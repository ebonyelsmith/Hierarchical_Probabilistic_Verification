import sys
from tqdm import tqdm
print(f"Python version: {sys.executable}")
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to use, e.g. "0", "0,1", "1", etc.
from dataclasses import dataclass, field
import pathlib
from typing import Optional, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from LCRL.env import DummyVectorEnv
from LCRL.exploration import GaussianNoise
from LCRL.data import Batch
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic
import LCRL.reach_rl_gym_envs as reach_rl_gym_envs

from env_utils import NoResetSyncVectorEnv, evaluate_V_batch, find_a_batch, find_a, get_args, get_env_and_policy
from env_utils_ppo import get_args_ppo, get_env_and_policy_ppo
from intent_estimation_utils import ControlGainEstimator
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm

from gymnasium.vector import SyncVectorEnv
from copy import deepcopy
from gymnasium.vector.utils import concatenate
# from mppi_mpc_controller import (
#     DroneMPPIConfig,
#     DroneMPPIController,
#     ReachabilityValueFunction,
# )
# from mpc_cbf_controller import DroneMPCConfig, DroneMPCCBFController

# from mppi_mpc_cbf_controller import (
#     MPPI_MPC_CBF_ControllerConfig,
#     MPPI_MPC_CBF_Controller,
# )

from scipy.interpolate import RegularGridInterpolator

# from local_verif_utils import get_beta5, beta, calibrate_V_vectorized, calibrate_V_scenario_local_vectorized, grow_regions_closest_point, grow_regions_closest_point_new, make_new_env, compute_min_scenarios_alex
from time import time

### Import Controller Classes ###
# local verif controller
# from controllers.local_verif_switch import ComputingVerifiedReachableSet, DroneRaceSimulation
# from controllers.local_verif_switch_updated import ComputingVerifiedReachableSet, DroneRaceSimulation
from controllers.local_verif_switch_updated_scen import ComputingVerifiedReachableSet, DroneRaceSimulation

# MPPI baseline (safe and non-safe)
from controllers.mppi_baseline import DroneRaceMPPIBaselineSimulation

# MPPI + CBF baseline
from controllers.mppi_cbf_baseline import DroneRaceMPPIBaselinewithCBFSimulation

# MPPI + safe cost + warm-starting with learned policy baseline
from controllers.mppi_warmstart_learned import DroneRaceMPPIBaselineWarmstartLearnedPolicySimulation

# Switching controller without learned policy in switching logic (ablation)
from controllers.switch_no_learned_policy import DroneRaceSimulationSwitchingNoLearnedPolicy

# Learned policy only baseline
from controllers.learned_policy_only import DroneRaceLearnedPolicyBaselineSimulation

# PPO + CBF baseline
from controllers.ppo_cbf import DroneRacePPOCBFSimulation

# set random seeds for reproducibility
seed = 15 #14 #13 #12 #11 #0
np.random.seed(seed)
torch.manual_seed(seed)


@dataclass
class DroneMPPIConfig:
    """Configuration container for the drone MPPI controller."""

    horizon: int = 15 #25 # Ebonye 1/27/2026
    dt: float = 0.1
    num_samples: int = 500
    temperature: float = 10.0 #1.0
    noise_sigma: Sequence[float] = (0.4, 0.4, 0.4)
    position_weight: float = 5.0
    velocity_weight: float = 1.0
    control_weight: float = 0.1
    safety_radius: float = 0.2
    safety_weight: float = 5.0
    value_weight: float = 1000000.0
    value_threshold: float = -0.02
    control_gain: float = 0.5 #2
    opponent_gain: float = 0.5 #1.0
    disturbance_gain: float = 0.1
    controlled_agent_index: int = 0
    num_agents: int = 2 #3 Ebonye 1/26/2026
    per_agent_state_dim: int = 6
    per_agent_control_dim: int = 3
    viability_state_perm: Sequence[int] = (
        0,
        1,
        2,
        3,
        4,
        5,
    )
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        sigma = np.asarray(self.noise_sigma, dtype=np.float64)
        if sigma.shape != (self.per_agent_control_dim,):
            raise ValueError(
                f"noise_sigma must be length-{self.per_agent_control_dim} for per-agent control."
            )
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not 0 <= self.controlled_agent_index < self.num_agents:
            raise ValueError("controlled_agent_index must select a valid agent.")
        perm = np.asarray(self.viability_state_perm, dtype=np.int64)
        if perm.shape[0] != self.per_agent_state_dim or sorted(perm.tolist()) != list(range(self.per_agent_state_dim)):
            raise ValueError("viability_state_perm must be a permutation of agent state indices.")
        self.noise_sigma = sigma
        self.viability_state_perm = tuple(int(x) for x in perm.tolist())

@dataclass
class MPPI_MPC_CBF_ControllerConfig:
    mppi_cfg: DroneMPPIConfig
    gamma_cbf: float = 0.6
    max_control: float = 1.0
    solver: str = "OSQP"
    solver_options: Dict[str, float] = field(default_factory=dict)
    filter_weight: Sequence[float] = (1.0, 1.0, 1.0)  # weight for the CBF safety filter in the optimization objective
    filter_slack_weight: float = 100000.0
    position_weight: Sequence[float] = (5.0, 5.0, 5.0)
    velocity_weight: Sequence[float] = (1.0, 1.0, 1.0)
    control_weight: Sequence[float] = (0.1, 0.1, 0.1)
    terminal_position_weight: Sequence[float] = (10.0, 10.0, 10.0)
    terminal_velocity_weight: Sequence[float] = (2.0, 2.0, 2.0)


    def __post_init__(self) -> None:
        self.max_control = self._to_weight(self.max_control, name="max_control", allow_scalar=True)
        self.filter_weight = self._to_weight(self.filter_weight, name="filter_weight", allow_scalar=True)
        self.filter_slack_weight = float(self.filter_slack_weight)
        if self.filter_slack_weight < 0.0:
            raise ValueError("filter_slack_weight must be non-negative")
        self.filter_slack_weight = float(self.filter_slack_weight)

    def _to_weight(
        self,
        value: float | Sequence[float],
        *,
        name: str,
        allow_scalar: bool = False,
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            if not allow_scalar:
                raise ValueError(f"{name} must have three elements")
            arr = np.repeat(float(arr), self.mppi_cfg.per_agent_control_dim)
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must contain exactly three entries")
        if np.any(arr < 0):
            raise ValueError(f"{name} entries must be non-negative")
        return arr
    
@dataclass
class DroneRaceConfig:
    """Configuration for the drone racing simulation."""

    # ego_speed_scale: float = 0.5
    # other_speed_scale: float = 1.0
    duration: float = 8
    # initial_state: np.ndarray = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    )
    save_path: pathlib.Path = pathlib.Path("experiment_script/data/drone_race_mppi.npz")
    value_path: Optional[pathlib.Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise an MPPI drone trajectory.")
    parser.add_argument(
        "--value-path",
        type=pathlib.Path,
        default=None,
        help="Optional reachability policy (policy.pth) for value-function penalties.",
    )
    parser.add_argument(
        "--controlled-agent",
        type=int,
        default=0,
        help="Index of the agent controlled by MPPI (default: 0).",
    )
    parser.add_argument(
        "--save-figure",
        type=pathlib.Path,
        default=None,
        help="If provided, path to save the figure instead of showing it interactively.",
    )
    parser.add_argument(
        "--save-gif",
        type=pathlib.Path,
        default=None,
        help="If provided, path to save an animated GIF of the rollout.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DroneRaceConfig.duration,
        help="Simulation duration in seconds (default: 12).",
    )
    
    parser.add_argument(
        "--save-data",
        type=pathlib.Path,
        default=None,
        help="Optional path to save the rollout logs (npz). Defaults to MPCRaceConfig.save_path if omitted.",
    )
    
    return parser.parse_args()



def main6() -> None:
    # This main function will conduct a monte carlo simulation to evaluate the performance of
    # Hybrid Controller, MPPI baseline, and MPPI with CBF across a range of initial conditions
    # with fixed opponent behavior. Trajectories will be saved to a file for later analysis and plotting.
    args = get_args()
    args2 = parse_args()
    args_ppo = get_args_ppo()
    env, policy_function = get_env_and_policy(args)
    env_ppo, ppo_policy = get_env_and_policy_ppo(args_ppo, epoch_id=100)

    # Define a set of initial conditions for the ego agent (sample in all 6 state dimensions)
    num_initial_conditions = 500
    ego_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    # ego_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    ego_vx_init = np.zeros(num_initial_conditions)  
    ego_y_init = np.random.uniform(-2.5, -1.0, size=num_initial_conditions) # 1.5 before 3/11/2026
    # ego_vy_init = np.random.uniform(0.7, 1.0, size=num_initial_conditions)
    ego_vy_init = np.random.uniform(0.5, 0.7, size=num_initial_conditions)  # want ego to be faster than adversary for sake of target set relevance
    ego_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    ego_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    ad_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    # ad_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    ad_vx_init = np.zeros(num_initial_conditions)
    ad_y_init = np.random.uniform(-2.5, -1.0, size=num_initial_conditions) # 1.5 before 3/11/2026
    # ad_vy_init = np.random.uniform(0.4, 0.7, size=num_initial_conditions)  # want ego to be faster than adversary for sake of target set relevance
    # randomly sample all adversary initial velocities to be slower than ego to ensure relevance of target set and safety set computations
    ad_vy_init = np.random.uniform(0.3, 0.5, size=num_initial_conditions)
    ad_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    ad_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    # Create a list of initial states for the simulations
    initial_states = np.stack([ego_x_init, ego_vx_init, ego_y_init, ego_vy_init, ego_z_init, ego_vz_init,
                                ad_x_init, ad_vx_init, ad_y_init, ad_vy_init, ad_z_init, ad_vz_init], axis=1)
    
    # Create a file path for saving the monte carlo results
    # monte_carlo_save_path = "monte_carlo_results_additional.npz" 
    # monte_carlo_save_path = "monte_carlo_results_no_learned_policy_ablation.npz"
    monte_carlo_save_path = f"monte_carlo_results_all_controllers_seed_{seed}_fastopp.npz"


    # Loop through initial states and run simulations for each controller type, saving trajectories to file
    # from tqdm import tqdm
    all_data_hybrid = []
    all_data_mppi_baseline = []
    all_data_mppi_cbf = []
    all_data_mppi_safe_baseline = []
    all_data_mppi_warmstart_learned_baseline = []
    all_data_policy_only_baseline = []
    all_data_hybrid_no_learned_policy = []
    all_data_ppo_cbf_baseline = []
    for initial_state in tqdm(initial_states, desc="Running simulations"):
    # for initial_state in initial_states:
        # Run Hybrid Controller (same as main2)
        # print("-" * 50)
        print(f"Running simulation with initial state: {initial_state}")
        race_config = DroneRaceConfig(
            duration=args2.duration,
            initial_state=initial_state,
            value_path=args2.value_path,
        )
        mppi_config = DroneMPPIConfig(controlled_agent_index=args2.controlled_agent)
        # mppi_config.opponent_gain = 0.5
        mppi_config.opponent_gain = 1.0 ## 3/11/2026: increasing opponent gain
        mppi_config.control_gain = 0.5
        verif_reach_set_computer = ComputingVerifiedReachableSet(
            current_state=initial_state,
        )
        offline_verified_set, _ = verif_reach_set_computer.compute_verified_set(
            args,
            deepcopy(mppi_config),
            policy_function,
            confidence=0.9,
            delta=1e-3
        )
        # # Run switching controller simulation
        sim_hybrid = DroneRaceSimulation(
            args=args,
            initial_state=initial_state,
            offline_verified_set=offline_verified_set,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
            reachability_value_path=args2.value_path,
        )
        data_hybrid = sim_hybrid.run()
        all_data_hybrid.append(data_hybrid)

        # import pdb; pdb.set_trace()

        ####

        # Run switching controller without learned policy in switching logic (ablation)
        ## debugging
        # initial_state_debug = np.array([-0.31192528, -0.18630495, -1.49892508,  0.80690527,  0.07623354,  0.03269144, 
        #                                 -0.60761344, -0.02786871, -0.75642083,  0.57885821,  0.02777298,  0.00635087])
        # initial_state_debug = np.array([0.28401337,  0.    ,     -2.48426176,  0.73743893,  0.09597282 , 0.02372815, 0.48932932,  0.     ,    -1.51672868 , 0.68517966  ,0.05612783,  0.05792003])
        sim_hybrid_no_learned_policy = DroneRaceSimulationSwitchingNoLearnedPolicy(
            args=args,
            initial_state=initial_state,
            offline_verified_set=offline_verified_set,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
            reachability_value_path=args2.value_path,
        )
        data_hybrid_no_learned_policy = sim_hybrid_no_learned_policy.run()
        all_data_hybrid_no_learned_policy.append(data_hybrid_no_learned_policy)
        # import pdb; pdb.set_trace()

        ####

        # Run MPPI baseline no safety
        sim_mppi_baseline = DroneRaceMPPIBaselineSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
        )
        data_mppi_baseline = sim_mppi_baseline.run()
        all_data_mppi_baseline.append(data_mppi_baseline)

        ####
        
        # Run MPPI baseline with safety in cost
        sim_mppi_baseline_safe = DroneRaceMPPIBaselineSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
            safe=True,
        )
        # sim_mppi_baseline_safe.mppi_cfg.safety_weight = 5.0
        data_mppi_baseline_safe = sim_mppi_baseline_safe.run()
        all_data_mppi_safe_baseline.append(data_mppi_baseline_safe)

        # Run MPPI with CBF
        mppi_cbf_cfg = MPPI_MPC_CBF_ControllerConfig(
            mppi_cfg = deepcopy(mppi_config),
        )
        sim_mppi_cbf = DroneRaceMPPIBaselinewithCBFSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cbf_cfg=mppi_cbf_cfg,
        )
        data_mppi_cbf = sim_mppi_cbf.run()
        all_data_mppi_cbf.append(data_mppi_cbf)

        # Run MPPI warm-started with learned policy
        sim_mppi_warmstart_learned = DroneRaceMPPIBaselineWarmstartLearnedPolicySimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
            # policy_function=policy_function,
            reachability_value_path=args2.value_path,
        )
        data_mppi_warmstart_learned = sim_mppi_warmstart_learned.run()
        all_data_mppi_warmstart_learned_baseline.append(data_mppi_warmstart_learned)

        # Run policy-only baseline (no MPC, just execute learned policy)
        sim_policy_only = DroneRaceLearnedPolicyBaselineSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
            # policy_function=policy_function,
            reachability_value_path=args2.value_path,
        )
        data_policy_only = sim_policy_only.run()
        all_data_policy_only_baseline.append(data_policy_only)

        # Run PPO + CBF baseline
        sim_ppo_cbf = DroneRacePPOCBFSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cbf_cfg=mppi_cbf_cfg,
            ppo_policy=ppo_policy,
        )
        data_ppo_cbf = sim_ppo_cbf.run()
        all_data_ppo_cbf_baseline.append(data_ppo_cbf)


        # Save intermediate results to file every 10 simulations
        if len(all_data_hybrid_no_learned_policy) % 10 == 0:
            np.savez(monte_carlo_save_path,
                        hybrid=all_data_hybrid, 
                        hybrid_no_learned_policy=all_data_hybrid_no_learned_policy,
                        mppi_baseline=all_data_mppi_baseline, 
                        mppi_safe_baseline=all_data_mppi_safe_baseline, 
                        mppi_cbf=all_data_mppi_cbf, 
                        mppi_warmstart_learned_baseline=all_data_mppi_warmstart_learned_baseline,
                        policy_only_baseline=all_data_policy_only_baseline,
                        ppo_cbf_baseline=all_data_ppo_cbf_baseline,
                        )



if __name__ == "__main__":
    main6()
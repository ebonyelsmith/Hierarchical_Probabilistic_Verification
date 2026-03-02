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
from intent_estimation_utils import ControlGainEstimator
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm

from gymnasium.vector import SyncVectorEnv
from copy import deepcopy
from gymnasium.vector.utils import concatenate
from mppi_mpc_controller import (
    DroneMPPIConfig,
    DroneMPPIController,
    ReachabilityValueFunction,
)
from mpc_cbf_controller import DroneMPCConfig, DroneMPCCBFController

from mppi_mpc_cbf_controller import (
    MPPI_MPC_CBF_ControllerConfig,
    MPPI_MPC_CBF_Controller,
)

from scipy.interpolate import RegularGridInterpolator

from local_verif_utils import get_beta5, beta, calibrate_V_vectorized, calibrate_V_scenario_local_vectorized, grow_regions_closest_point, make_new_env, compute_min_scenarios_alex
from time import time

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

    
class DroneRaceLearnedPolicyBaselineSimulation:
    """Simulates an ego drone trying to overtake another drone using a learned policy baseline without any MPPI or verified set."""

    def __init__(
            self,
            args,
            initial_state: np.ndarray,
            sim_cfg: DroneRaceConfig,
            mppi_cfg: DroneMPPIConfig,
            # policy_function: ReachabilityValueFunction
            reachability_value_path: Optional[pathlib.Path] = None
    ) -> None:
        self.sim_cfg = sim_cfg
        # self.policy_function = policy_function.policy
        value_fn = (
            ReachabilityValueFunction.from_policy_path(
                str(reachability_value_path),
                device="cpu",
            )
            if reachability_value_path is not None
            else None
        )
        self.policy_function = value_fn.policy
        self.mppi_cfg = mppi_cfg
        self.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the learned policy baseline
        self.mppi_cfg.control_gain = 0.5  # assume a fixed control gain for the learned policy baseline
        self.mppi_controller = DroneMPPIController(mppi_cfg)  # only used for simulate_step, not for control in this baseline
        self.dt = 0.1  # assume same time step as MPPI for fair comparison
        self.num_steps = int(np.ceil(sim_cfg.duration / self.dt))
        self.state = initial_state
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, 3), dtype=np.float64)  # assume control dimension of 3

    def find_a(self, state):
            tmp_obs = np.array(state).reshape(1,-1)
            tmp_batch = Batch(obs = tmp_obs, info = Batch())
            tmp = self.policy_function(tmp_batch, model = "actor_old").act
            act = self.policy_function.map_action(tmp).cpu().detach().numpy().flatten()
            return act

    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""
        for t in range(self.num_steps):
            # reset_nominal = False
            action = self.find_a(self.state)
            reached_goal = self.state[2] > 0.0 and np.abs(self.state[0]) <= 0.3
            if reached_goal:
                self.state_log[t:] = self.state  # log current state at the time of reaching goal
                self.control_log[t:] = action[:3]  # log control input at the time of reaching goal
                break
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            next_state, _ = self.mppi_controller.simulate_step(self.state, action)
            self.state = next_state

        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
        }
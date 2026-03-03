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

from local_verif_utils import get_beta5, beta, calibrate_V_vectorized, calibrate_V_scenario_local_vectorized, grow_regions_closest_point, make_new_env, compute_min_scenarios_alex, grow_regions_closest_point_new
from time import time

@dataclass
class DroneMPPIConfig:
    """Configuration container for the drone MPPI controller."""

    horizon: int = 15 #25 # Ebonye 1/27/2026
    dt: float = 0.1
    num_samples: int = 1000
    temperature: float = 10.0 #1.0
    noise_sigma: Sequence[float] = (0.4, 0.4, 0.4)
    position_weight: float = 5.0
    velocity_weight: float = 2.0
    control_weight: float = 0.1
    safety_radius: float = 0.2
    safety_weight: float = 5.0
    value_weight: float = 1000000.0
    value_threshold: float = -0.02
    control_gain: float = 2
    opponent_gain: float = 1.0
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


class DroneMPPIControllerLocalVerif(DroneMPPIController):
    """MPPI controller that can use a local verified reachable set for safety."""

    def __init__(
        self,
        config: DroneMPPIConfig,
        value_function: Optional[ReachabilityValueFunction] = None,
    ):
        super().__init__(config, value_function)
        self.closest_safe_point = None  # to store closest safe point for local verification

    def _stage_cost(
        self,
        state: np.ndarray,
        control: np.ndarray,
        t: int,
        value: Optional[float],
    ) -> Tuple[float, float]:
        
        # pos_error = self._agent_position(state, self.controlled_agent) - self._ref_positions[t]
        ref_pos = self.closest_safe_point if self.closest_safe_point is not None else self._ref_positions[t]
        pos_error = self._agent_position(state, self.controlled_agent) - ref_pos
        vel_error = self._agent_velocity(state, self.controlled_agent) - self._ref_velocities[t]

        cost = (
            self.config.position_weight * float(np.dot(pos_error, pos_error))
            + self.config.velocity_weight * float(np.dot(vel_error, vel_error))
            + self.config.control_weight * float(np.dot(control, control))
        )

        # 1/27/2026 (ebonye): augment stage cost with reach-avoid value function as a terminal cost
        # drone racing literature: model based (mpc), pure end-to-end (ppo), combination (train ppo to check trajectory)

        if value is not None:
            violation = self.config.value_threshold - value
            if violation > 0.0:
                cost += self.config.value_weight * violation * violation

        margin = self._safety_margin(state)
        if margin < 0.0:
            cost += self.config.safety_weight * margin * margin
        return cost, margin

class DroneMPPIControllerFast(DroneMPPIController):
    """MPPI controller with a reference trajectory that maintains high speed through the gate."""

    def __init__(
        self,
        config: DroneMPPIConfig,
        value_function: Optional[ReachabilityValueFunction] = None,
    ):
        super().__init__(config, value_function)

    def _stage_cost(
        self,
        state: np.ndarray,
        control: np.ndarray,
        t: int,
        value: Optional[float],
    ) -> Tuple[float, float]:
        
        x, y, z = self._agent_position(state, self.controlled_agent)
        vx, vy, vz = self._agent_velocity(state, self.controlled_agent)
        pos_error = self._agent_position(state, self.controlled_agent) - self._ref_positions[t]
        vel_error = self._agent_velocity(state, self.controlled_agent) - self._ref_velocities[t]
        lateral_priority = [1.0, 0.0, 1.0]
        longitudinal_priority = [0.0, 1.0, 0.0]

        cost = (
            self.config.position_weight * float(np.dot(pos_error * lateral_priority, pos_error * lateral_priority)) * 5000.0
            + self.config.velocity_weight * float(np.dot(vel_error * longitudinal_priority, vel_error * longitudinal_priority)) * 1.0
            # + self.config.control_weight * float(np.dot(control, control)) * 0.1
        )
        vel_cost = 0.0
        if x * vx > 0.0:
            vel_cost += 50000.0 * (vx**2)
        if z * vz > 0.0:
            vel_cost += 50000.0 * (vz**2)
        cost += vel_cost

        ctrl_cost = self.config.control_weight * float(np.dot(control, control)) * 0.01
        cost += ctrl_cost




        boundary = 0.15
        wall_penalty = 0.0
        if abs(x) > boundary or abs(z) > boundary:
            wall_penalty = 50000.0 * (max(abs(x), abs(z)) - boundary + 0.1)**2
        
        cost += wall_penalty
        if t == self.config.horizon - 1:
            cost += 10000.0 * (x**2 + z**2)  # strong terminal cost to encourage reaching the end of the reference trajectory and not going too far ahead
        # margin = self._safety_margin(state)
        # if margin < 0.0:
        #     cost += self.config.safety_weight * margin * margin
        return cost, 0.0






class VerifiedReachableSet:
    """Class to represent a verified reachable set using a grid-based value function."""

    def __init__(
        self,
        grid_axes: Sequence[np.ndarray],
        value_function: np.ndarray,
    ):
        self.grid_axes = grid_axes
        self.value_function = value_function
        self.interpolator = RegularGridInterpolator(
            points=grid_axes,
            values=value_function,
            bounds_error=False,
            fill_value=np.min(value_function) - 1.0,
        )
    
    def is_inside(self, state: np.ndarray) -> bool:
        """Check if the given state is inside the verified reachable set."""
        # value = self.interpolator(state)
        value = self.interpolator(state[[1, 0]])  # note the order of state variables to match grid axes
        # print(f"Verified value function at state {state}: {value}")
        return value > 0.0
    
    def find_closest_safe_point(self, current_ego_state: np.ndarray) -> np.ndarray:
        """
        Find the (x, y) coordinates in the grid that are safe (value > 0) and closest to the current ego state.
        """
        # Get all grid coordinates where value is safe
        # Indices are (y_idx, x_idx) because grid is [y, x]
        safe_y_indices, safe_x_indices = np.where(self.value_function > 0)

        if len(safe_x_indices) == 0:
            return None  # No safe points available
        
        # Map indices back to real world coordinates
        safe_y_coords = self.grid_axes[0][safe_y_indices]
        safe_x_coords = self.grid_axes[1][safe_x_indices]
        safe_points = np.column_stack((safe_x_coords, safe_y_coords))

        # Compute distances from current ego state to all safe points
        ego_xy = current_ego_state[[0, 2]]  # extract (x, y) from state
        distances = np.linalg.norm(safe_points - ego_xy, axis=1)

        # Return coordinate with minimum distance
        closest_idx = np.argmin(distances)
        closest_safe_point = safe_points[closest_idx]
        return closest_safe_point


class ComputingVerifiedReachableSet:
    """Class to compute the verified reachable set using Lipschitz constants and sampling."""

    def __init__(
        self,
        current_state: np.ndarray,
        Lf: float = 1.05125,
        Lc: float = 20,
        Lr: float = 10,
        epsilon_x: float = 0.1,
        reachability_horizon: int = 30,
        gamma: float = 0.95,
        alphaC_list = None,
        alphaR_list = None,
        alphaC_scenario_list = None,
        alphaR_scenario_list = None,

    ):
        self.current_state = current_state
        self.Lf = Lf
        self.Lc = Lc
        self.Lr = Lr
        self.epsilon_x = epsilon_x
        self.reachability_horizon = reachability_horizon
        self.gamma = gamma
        self.alphaC_list = alphaC_list
        self.alphaR_list = alphaR_list
        self.alphaC_scenario_list = alphaC_scenario_list
        self.alphaR_scenario_list = alphaR_scenario_list


    def compute_verified_set(
        self,
        args,
        mppi_cfg: DroneMPPIConfig,
        policy_function: ReachabilityValueFunction,
        confidence: float = 0.97,
        delta: float = 1e-3) -> VerifiedReachableSet:

        """Compute the verified reachable set using sampling and Lipschitz constants."""

        if self.alphaC_scenario_list is None or self.alphaC_list is None:
            self.alphaC_scenario_list, self.alphaR_scenario_list, self.alphaC_list, self.alphaR_list = np.zeros(self.reachability_horizon), np.zeros(self.reachability_horizon), np.zeros(self.reachability_horizon), np.zeros(self.reachability_horizon)
            env = make_new_env(args)
            env.control_gain_2 = mppi_cfg.opponent_gain 
            # scenario_radii3, initial_states3, nominal_trajs3, state_trajs3, betas3 = get_beta5(
            #     env,
            #     policy_function,
            #     self.reachability_horizon,
            #     self.epsilon_x,
            #     0,
            #     args,
            #     self.gamma,
            #     confidence,
            #     delta
            # )
    
            for t in range(self.reachability_horizon):
                # self.alphaC_scenario_list[t] = self.Lc*betas3[t]
                # self.alphaR_scenario_list[t] = self.Lr*betas3[t]
                self.alphaC_list[t] = self.Lc*beta(self.reachability_horizon, self.Lf, 0, self.epsilon_x, 0, self.gamma)
                self.alphaR_list[t] = self.Lr*beta(self.reachability_horizon, self.Lf, 0, self.epsilon_x, 0, self.gamma)

        # Define grid for state space (x, y)
        x = np.arange(-0.9, 0.9, self.epsilon_x)
        y = np.arange(-2.6, 0, self.epsilon_x)
        X, Y = np.meshgrid(x, y)
        H, W = X.shape
        tmp_states = np.empty((H, W, 12))
        tmp_states[:, :, 0] = X
        # import pdb; pdb.set_trace()
        # print(f"self.current_state: {self.current_state}")
        tmp_states[:, :, 1] = self.current_state[1]
        tmp_states[:, :, 2] = Y
        tmp_states[:, :, 3:] = self.current_state[3:]
        tmp_states_reshaped = tmp_states.reshape(-1, 12)

        # print(f"alphaC_list: {self.alphaC_list}")
        env = make_new_env(args)
        env.control_gain_2 = mppi_cfg.opponent_gain

        # print(f"policy_function: {policy_function}")
        V_lp_vectorized_flat, _, _ = calibrate_V_vectorized(
            env,
            policy_function,
            tmp_states_reshaped,
            self.reachability_horizon,
            self.alphaC_list,
            self.alphaR_list,
            args,
            self.gamma
        )
        V_lp_vectorized = V_lp_vectorized_flat.reshape(H, W)


        # import pdb; pdb.set_trace()
        verified_set_deterministic = VerifiedReachableSet(
            # grid_axes=[x, y],
            grid_axes=[y, x],
            value_function=V_lp_vectorized,
        )

        # verified_set_scenario = VerifiedReachableSet(
        #     # grid_axes=[x, y],
        #     grid_axes=[y, x],
        #     value_function=V_lp_scenario_vectorized,
        # )
        return verified_set_deterministic, None #verified_set_scenario
   

class SwitchingDroneController:
    """
    High-level controller that switches between:
      1. DDPG overtaking policy (when inside Backward Reachable Set)
      2. MPPI (outside of BRS and trying to reach locally approximated BRS)
      3. High-speed maintain lane (when safely ahead of opponent)
    """
    def __init__(
        self,
        args,
        num_steps: int,
        mppi_cfg: DroneMPPIConfig,
        policy_function: ReachabilityValueFunction,
        verified_reachable_set: VerifiedReachableSet,
    ):
        self.args = args
        self.num_steps = num_steps
        self.mppi_cfg = deepcopy(mppi_cfg)
        self.mppi_cfg.horizon = 20  # increase horizon for better performance in the local MPPI mode
        self.mppi_fast_cfg = deepcopy(mppi_cfg)
        self.mppi_fast_cfg.velocity_weight = 0.5  # increase velocity weight for high-speed lane maintain mode
        self.mppi_fast_cfg.horizon = 20  # increase horizon for high-speed lane maintain mode to encourage more foresight in maintaining high speed
        # self.mppi_controller = DroneMPPIController(self.mppi_cfg)
        self.mppi_controller_local = DroneMPPIControllerLocalVerif(self.mppi_cfg, value_function=policy_function)
        # self.mppi_controller_fast = DroneMPPIController(self.mppi_fast_cfg) #s, value_function=policy_function)
        self.mppi_controller_fast = DroneMPPIControllerFast(self.mppi_fast_cfg)
        # self.mppi_controller_fast_near_gate = DroneMPPIController(self.mppi_fast_cfg)
        self.mppi_controller_fast_near_gate = DroneMPPIControllerFast(self.mppi_fast_cfg)
        self.mppi_controller_simulate_step = DroneMPPIController(self.mppi_cfg, value_function=policy_function)
        self.policy_function = policy_function.policy
        self.verified_reachable_set = verified_reachable_set
        # self.target_set_reached = False
        self.Lf = 1.05125 
        self.Lc = 20
        self.Lr = 10
        self.epsilon_x = 0.1
        self.reachability_horizon = self.num_steps #30  # steps
        self.gamma = 0.95
        self.verified_reachable_set_computer = ComputingVerifiedReachableSet(
            current_state=None,
            Lf=self.Lf,
            Lc=self.Lc,
            Lr=self.Lr,
            epsilon_x=self.epsilon_x,
            reachability_horizon=self.reachability_horizon,
            gamma=self.gamma,
        )
        self.recompute = True  # flag to indicate if we need to recompute local verified set
        self.recompute_local = True  # flag to indicate if we need to recompute local growth set
        self.is_safe_local = False
        self.expanded_region = None
        self.reached_goal = False
        self.near_gate = False
        self.is_in_target_set = False
        self.first_time_in_target_set = False
        self.target_set_modified = None
        self.terminate = False


    def find_a(self,state):
            tmp_obs = np.array(state).reshape(1,-1)
            tmp_batch = Batch(obs = tmp_obs, info = Batch())
            # tmp = self.policy_function.policy(tmp_batch, model = "actor_old").act
            # act = self.policy_function.policy.map_action(tmp).cpu().detach().numpy().flatten()
            tmp = self.policy_function(tmp_batch, model = "actor_old").act
            act = self.policy_function.map_action(tmp).cpu().detach().numpy().flatten()
            
            return act
    
    def generate_learned_policy_reference(
        self,
        policy_function: ReachabilityValueFunction,
        start_state: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """Generate reference states by rolling out a learned policy from a start state."""

        def find_a(state):
            tmp_obs = np.array(state).reshape(1,-1)
            tmp_batch = Batch(obs = tmp_obs, info = Batch())
            tmp = policy_function(tmp_batch, model = "actor_old").act
            act = policy_function.map_action(tmp).cpu().detach().numpy().flatten()
            return act


        state_dim = start_state.shape[0]
        # per_agent_state_dim = policy_function.per_agent_state_dim
        per_agent_state_dim = self.mppi_cfg.per_agent_state_dim
        if state_dim % per_agent_state_dim != 0:
            raise ValueError("start_state has incompatible dimension with policy_function.")
        num_agents = state_dim // per_agent_state_dim
        reference = np.zeros((num_points, per_agent_state_dim), dtype=np.float64)
        reference_actions = np.zeros((num_points, self.mppi_cfg.per_agent_control_dim), dtype=np.float64)
        current_state = start_state.copy()
        cnfg = DroneMPPIConfig(num_agents=1)
        cntrllr = DroneMPPIController(cnfg, value_function=policy_function)  
        # print(f"num_agents = {cntrllr.num_agents}")  
        reference[0] = current_state[:per_agent_state_dim]

        # print(f"num_points: {num_points}")
        # from tqdm import tqdm
        for t in range(num_points - 1):
        # for t in tqdm(tqdm(range(num_points-1))):
            # controlled_sl = slice(
            #     policy_function.controlled_agent * per_agent_state_dim,
            #     (policy_function.controlled_agent + 1) * per_agent_state_dim,
            # )
            # reference[t] = current_state[controlled_sl]

            action = find_a(current_state)[:3]  # extract control for controlled agent
            next_state, _ = cntrllr.simulate_step(current_state, action)
            current_state = next_state
            reference[t+1] = current_state[:per_agent_state_dim]
            reference_actions[t] = action
        return reference, reference_actions
    
    
    
    def generate_sliding_track_reference(
        self,
        state: np.ndarray,
        num_points: int,
        target_speed: float,
        dt: float = 0.1,
    ) -> np.ndarray:
        curr_y = state[2]

        reference = np.zeros((num_points, 6), dtype=np.float64)
        for i in range(num_points):
            ref_y = curr_y + target_speed * dt * i
            ref_x = 0.0
            ref_z = 0.0

            # if ref_y > 1.0 and state[2] > -0.1:
            #     ref_y = 1.0  # cap at gate line to avoid going too far ahead
            
            # if ref_y > 0.0: #and state[2] <= -0.1:
            #     ref_y = 0.0

            ref_vy = target_speed
            ref_vx = 0.0
            ref_vz = 0.0

            reference[i] = np.array([ref_x, ref_vx, ref_y, ref_vy, ref_z, ref_vz])
        
        return reference
    
    
    def reset(self):
        self.recompute = True
        self.recompute_local = True
        self.is_safe_local = False
        self.expanded_region = None
        self.reached_goal = False
        self.near_gate = False
        self.is_in_target_set = False
        self.first_time_in_target_set = False
        self.target_set_modified = None
        self.terminate = False

    def solve(
        self,
        state: np.ndarray,
        previous_action: Optional[np.ndarray] = None,
        reset_nominal: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """Return control action for the ego drone given the current state."""

        # Evaluate the verified value function at current state to decide if safe
        ego_xy = state[[0, 2]]
        if verbose:
            print(f"Current state: {state}")
            print(f"ego_xy: {ego_xy}")
        # is_safe = self.verified_reachable_set.is_inside(ego_xy)
        func_scale = 10.0
        rew = func_scale*min([
            state[2] - state[8],
            state[3] - state[9],
            (state[0] - -0.3),
            (0.3 - state[0]),
            (state[4] - -0.3),
            (0.3 - state[4])
        ])
        is_in_target_set = rew > 0

        if not self.is_in_target_set and is_in_target_set:
            self.is_in_target_set = True
            self.first_time_in_target_set = True
        

        if verbose:
            print(f"is_in_target_set: {self.is_in_target_set}")
            print(f"self.expanded_region: {self.expanded_region}")

        near_gate = np.linalg.norm(state[2]-0.0) <= 0.1 and np.abs(state[0]) <= 0.3
        # self.near_gate = near_gate

        if verbose:
            print(f"near_gate: {near_gate}")

        # if self.near_gate:
        #     print("Near gate!")

        if not self.reached_goal and state[2] > 0.0 and np.abs(state[0]) <= 0.3:
            self.reached_goal = True
            # self.reset()
            if verbose:
                print("Reached goal!")
            return np.zeros(6), 2, None  # no control input, high-speed lane maintain mode, no expanded region
        
        if not self.reached_goal and state[2] > 0.0 and np.abs(state[0]) > 0.3:
            if verbose:
                print("Reached goal but outside of gate bounds, terminating episode.")
            self.terminate = True
            return np.zeros(6), 2, None  # no control input, high-speed lane maintain mode, no expanded region

        mode = 0 # 0: learned policy, 1: local verif + mppi, 2: high-speed lane maintain

        self.near_gate = near_gate
        # if not is_in_target_set:
        if not self.is_in_target_set and not self.reached_goal and not self.near_gate:
            if self.recompute:
                self.recompute_local = True  # also recompute local growth set when verified set is recomputed
                
                self.verified_reachable_set_computer.current_state = state
                
                self.verified_reachable_set_computer.Lf = self.Lf
                # print(f"self.policy_function: {self.policy_function}")
                verified_set_deterministic, _ = self.verified_reachable_set_computer.compute_verified_set(
                    self.args,
                    self.mppi_cfg,
                    self.policy_function,   
                )
                self.verified_reachable_set = verified_set_deterministic
                
                # import pdb; pdb.set_trace()  # update to new verified set
                # self.recompute = False  # reset flag after recomputing verified set
        
            is_safe = self.verified_reachable_set.is_inside(ego_xy)
            if verbose:
                print(f"is_safe: {is_safe}")
            # import pdb; pdb.set_trace()

            if is_safe:
                # Inside verified BRS (given new intent): use learned policy
                action = self.find_a(state)
                mode = 0
                action = [action[0], action[1], action[2], 0.0, 0.0, 0.0]  # zero angular rates
                return action, mode, self.expanded_region
            else:
                # Grow a local probabilistic verified reachable set on closest boundary point of global BRS
                env = make_new_env(self.args)
                env.control_gain_2 = self.mppi_cfg.opponent_gain
                # print(f"state: {state}")

                if self.recompute_local:
                    start_time = time()
                    x = np.arange(-0.9, 0.9, self.epsilon_x)
                    y = np.arange(-2.6, 0, self.epsilon_x)
                    X, Y = np.meshgrid(x, y)

                    confidence = 0.9
                    delta = 1e-3
                    n_samples = compute_min_scenarios_alex(1-confidence, delta, 12)
                    
                    # print(f"state: {state}")
                    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")
                    # expanded_region, _, _, _ = grow_regions_closest_point(
                    expanded_region, _, _, _ = grow_regions_closest_point_new(
                        # state[[0, 2]],
                        state,
                        # self.verified_reachable_set.value_function,
                        X,
                        Y,
                        env,
                        self.reachability_horizon,
                        self.verified_reachable_set_computer.alphaC_list,
                        self.verified_reachable_set_computer.alphaR_list,
                        self.policy_function,
                        self.args,
                        self.verified_reachable_set.value_function,
                        max_attept_radius = 0.55,
                        N_samples = n_samples,
                        tol=1e-2
                    )
                    self.expanded_region = expanded_region
                    x_center, y_center, r_safe = expanded_region
                    # print(f"state: {state} ")

                    # if r_safe == 0:
                    #     if verbose:
                    #         print("Warning: There is no zero-level set for the value function, local growth failed. Trying to grow from target set instead.")
                        
                    #     x = np.arange(-0.9, 0.9, self.epsilon_x)
                    #     y = np.arange(-2.6, 0, self.epsilon_x)
                    #     X, Y = np.meshgrid(x, y)
                    #     state_target = target_set(X, Y, state.copy())
                        
                    #     # expanded_region, _, _, _ = grow_regions_closest_point(
                    #     expanded_region, _, _, _ = grow_regions_closest_point_new(
                    #         # state[[0, 2]],
                    #         state,
                    #         # self.verified_reachable_set.value_function,
                    #         X,
                    #         Y,
                    #         env,
                    #         self.reachability_horizon,
                    #         self.verified_reachable_set_computer.alphaC_list,
                    #         self.verified_reachable_set_computer.alphaR_list,
                    #         self.policy_function,
                    #         self.args,
                    #         state_target,
                    #         max_attept_radius = 1.0, #0.5,
                    #         N_samples = n_samples,
                    #         tol=1e-2,
                    #         target=True)
                    #     x_center_target, y_center_target, r_safe_target = expanded_region
                    #     self.expanded_region = expanded_region
                       
                    # # x_center, y_center, r_safe = expanded_region
                    # self.expanded_region = expanded_region
                

                    # # import pdb; pdb.set_trace()
                    # # check if current state is inside local growth set
                    
                    # # self.recompute_local = False  # reset flag after recomputing local growth set
                    end_time = time()
                    # if verbose:
                    # print(f"Time taken to compute local growth set: {end_time - start_time:.2f} seconds")

                x_center, y_center, r_safe = self.expanded_region
                is_safe_local = np.linalg.norm(ego_xy - np.array([x_center, y_center])) <= r_safe
                self.is_safe_local = is_safe_local
                if verbose:
                    print(f"is_safe_local: {is_safe_local}")
                # if is_safe_local:
                if self.is_safe_local: #or r_safe == 0:
                    # inside local growth set: use learned policy
                    action = self.find_a(state)
                    mode = 0
                    return action, mode, self.expanded_region
                elif not self.is_safe_local and r_safe > 0:
                    # outside local growth set but there is a nontrivial local growth set: use MP
                    x_center, y_center, r_safe = self.expanded_region
                    center = np.array([x_center, y_center])
                    direction = ego_xy - center
                    direction_norm = np.linalg.norm(direction)

                    closest_safe_point = center + (r_safe / direction_norm) * direction
                    closest_safe_point = np.array([closest_safe_point[0], closest_safe_point[1], state[4]])  # keep z coordinate the same

                    self.mppi_controller_local.closest_safe_point = closest_safe_point
                    reference, reference_actions = self.generate_learned_policy_reference(
                        self.policy_function,
                        state,
                        self.mppi_cfg.horizon
                    )
                    self.mppi_controller_local.nominal_sequence = reference_actions
                    have_new_nominal = True
                    if reset_nominal and have_new_nominal:
                        reset_nominal = False
                    action, info = self.mppi_controller_local.solve(state, reference, reset_nominal)
                    mode = 1
                    return action, mode, self.expanded_region
                
                elif not self.is_safe_local and r_safe == 0:
                    # # no local growth set found: use MPPI towards closest point in target set without velocity constraint
                    # if verbose:
                    #     print(f"No local growth set found, using MPPI towards closest point in target set without velocity constraint.")
                    # x = np.arange(-0.9, 0.9, self.epsilon_x)
                    # y = np.arange(-2.6, 0, self.epsilon_x)
                    # X, Y = np.meshgrid(x, y)
                    # target_set_modified = target_set_last_resort(X, Y, state.copy())
                    # # self.target_set_modified = target_set_modified
                    # # find closest point in modified target set
                    # target_points = np.column_stack((X.flatten(), Y.flatten()))
                    # target_values = target_set_modified.flatten()
                    # valid_indices = np.where(target_values > 0)[0]
                    # if len(valid_indices) == 0:
                    #     if verbose:
                    #         print("Warning: Modified target set is empty, cannot generate reference. Using straight line reference to goal instead.")
                    #     closest_point = np.array([0.0, 0.0, state[4]])  # just use current x,y and keep velocity the same
                    # else:
                    #     ego_xy = state[[0, 2]]
                    #     diff = target_points[valid_indices] - ego_xy
                    #     distances = np.linalg.norm(diff, axis=1)
                    #     # print(f"state: {state}")
                    #     # print(f"len(distances): {len(distances)}")
                    #     # print(f"np.argmin(distances): {np.argmin(distances)}")
                    #     closest_xy = target_points[valid_indices[np.argmin(distances)]]
                    #     closest_point = np.array([closest_xy[0], closest_xy[1], 0.0])  # make z coordinate 0
                    
                    # No local growth set found: use MPPI towards closest point in the global verified reachable set
                    closest_xy = self.verified_reachable_set.find_closest_safe_point(state)
                    if closest_xy is None:
                        if verbose:
                            print("Warning: No safe point found in verified reachable set. Just try to reach the goal directly.")
                        closest_point = np.array([0.0, 0.0, 0.0])  # just use current x,y and keep velocity the same
                    else:
                        if verbose:
                            print(f"No local growth set found, using MPPI towards closest point in verified reachable set: {closest_xy}")
                        closest_point = np.array([closest_xy[0], closest_xy[1], state[4]])  # keep z coordinate the same

                    self.mppi_controller_local.closest_safe_point = closest_point
                    reference, reference_actions = self.generate_learned_policy_reference(
                        self.policy_function,
                        state,
                        self.mppi_cfg.horizon
                    )
                    self.mppi_controller_local.nominal_sequence = reference_actions
                    have_new_nominal = True
                    if reset_nominal and have_new_nominal:
                        reset_nominal = False
                    action, _ = self.mppi_controller_local.solve(state, reference, reset_nominal )
                    mode = 1
                    return action, mode, self.expanded_region
        
        # else:
        if (self.is_in_target_set and not self.reached_goal):
            # Safely ahead of opponent: maintain high speed along lane center using MPPI
            desired_speed = 0.5 #0.7  # m/s
            # desired_position = np.array([0.0, state[1], state[2]])  # keep current y, z positions
            x_star = self.mppi_controller_fast._x_star
            
            ###
            goal_state = x_star.copy()
            goal_state[2] = 0.5  # set goal y position to be just past the gate to encourage more aggressive behavior in passing through the gate
            # print(f"Goal state for high-speed lane maintain: {goal_state}")


            # print(f"state: {state}")
            reference = self.generate_sliding_track_reference(
                state,
                self.mppi_fast_cfg.horizon,
                desired_speed,
                dt=self.mppi_fast_cfg.dt,
            )
            
            if self.first_time_in_target_set:
                # if it's the first time we enter the target set, set nominal sequence to be tiled previous action to encourage smoother transition into high-speed lane maintain mode
                # nominal_sequence = np.tile(-previous_action if previous_action is not None else np.zeros(6), (self.mppi_fast_cfg.horizon, 1))
                vx = state[1]
                vy = state[3]
                brake_action = np.zeros(3)
                brake_action[0] = -np.sign(vx) * 1.0 if np.abs(vx) > 0.35 else 0.0
                brake_action[1] = -np.sign(vy) * 0.6 if np.abs(vx) > 0.35 else 0.0
                nominal_sequence = np.tile(brake_action, (self.mppi_fast_cfg.horizon, 1))
                self.mppi_controller_fast.nominal_sequence = nominal_sequence
                self.first_time_in_target_set = False
            # else:
            #     self.mppi_controller_fast.nominal_sequence = np.zeros((self.mppi_fast_cfg.horizon, 3))  # zero nominal sequence for MPPI to encourage following the reference closely
            action, _ = self.mppi_controller_fast.solve(state, reference, reset_nominal)
            mode = 2
            return action, mode, None

        elif self.near_gate:
            # near the gate but not yet passed it: use high-speed MPPI with a reference that goes through the gate to encourage aggressive behavior in passing through the gate
            desired_speed = 0.7 #0.5 #0.7  # m/s

            x_star = self.mppi_controller_fast_near_gate._x_star
            goal_state = x_star.copy()
            # goal_state[2] = 0.75  # set goal y position to be just past the gate to encourage more aggressive behavior in passing through the gate
            if verbose:
                print(f"Goal state for near gate high-speed MPPI: {goal_state}")
            
            reference = self.generate_sliding_track_reference(
                state,
                self.mppi_fast_cfg.horizon,
                desired_speed,
                dt=self.mppi_fast_cfg.dt,
            )
            
            # print(f"Generated straight line reference for near gate high-speed MPPI: {reference}")
            if self.first_time_in_target_set:
                # if it's the first time we enter the target set, set nominal sequence to be tiled previous action to encourage smoother transition into high-speed lane maintain mode
                # nominal_sequence = np.tile(-previous_action if previous_action is not None else np.zeros(6), (self.mppi_fast_near_gate_cfg.horizon, 1))
                vx = state[1]
                vy = state[3]
                brake_action = np.zeros(3)
                brake_action[0] = -np.sign(vx) * 1.0 if np.abs(vx) > 0.35 else 0.0
                brake_action[1] = -np.sign(vy) * 0.6 if np.abs(vx) > 0.35 else 0.0
                nominal_sequence = np.tile(brake_action, (self.mppi_fast_near_gate_cfg.horizon, 1))
                self.mppi_controller_fast_near_gate.nominal_sequence = nominal_sequence
                self.first_time_in_target_set = False
            action, _ = self.mppi_controller_fast_near_gate.solve(state, reference, reset_nominal)
            mode = 2
            return action, mode, None
        

def target_set(X, Y, tmp_point):
    reward = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tmp_point[0], tmp_point[2] = X[i,j], Y[i,j]
            func_scale = 10.0
            reward[i,j] = 1 if (func_scale*min([
                tmp_point[2] - tmp_point[8],
                tmp_point[3] - tmp_point[9],
                (tmp_point[0] - -0.3),
                (0.3 - tmp_point[0]),
                (tmp_point[4] - -0.3),
                (0.3 - tmp_point[4])])) >= 0 else 0
    return reward

def target_set_last_resort(X, Y, tmp_point):
    reward = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tmp_point[0], tmp_point[2] = X[i,j], Y[i,j]
            func_scale = 10.0
            reward[i,j] = 1 if (func_scale*min([
                tmp_point[2] - tmp_point[8],
                # tmp_point[3] - tmp_point[9],
                (tmp_point[0] - -0.3),
                (0.3 - tmp_point[0]),
                (tmp_point[4] - -0.3),
                (0.3 - tmp_point[4])])) >= 0 else 0
           
    return reward

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

class DroneRaceSimulation:
    """Simulates an ego drone trying to overtake another drone to reach the target first."""

    def __init__(
            self,
            args,
            initial_state: np.ndarray,
            offline_verified_set: VerifiedReachableSet,
            sim_cfg: DroneRaceConfig,
            mppi_cfg: DroneMPPIConfig,
            reachability_value_path: Optional[pathlib.Path] = None
            # reachability_policy
    ) -> None:
        self.sim_cfg = sim_cfg
        self.mppi_cfg = mppi_cfg
        # self.mppi_controller = DroneMPPIController(mppi_cfg)

        value_fn = (
            ReachabilityValueFunction.from_policy_path(
                str(reachability_value_path),
                device="cpu",
            )
            if reachability_value_path is not None
            else None
        )

        # value_fn = reachability_policy

        self.dt = self.mppi_cfg.dt
        self.num_steps = int(np.ceil(sim_cfg.duration / self.dt))
        self.controller = SwitchingDroneController(
            args=args,
            num_steps=self.num_steps,
            mppi_cfg=self.mppi_cfg,
            policy_function=value_fn,
            verified_reachable_set=offline_verified_set,            
        )
        self.state = initial_state
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, self.mppi_cfg.per_agent_control_dim), dtype=np.float64)
        self.mode_log = np.zeros((self.num_steps,), dtype=np.int32)
        self.control_gain_estimator = ControlGainEstimator(window_size=6, dt=self.dt)
        self.expanded_region_log = []

    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""
        

        for t in range(self.num_steps):
        # for t in tqdm(range(self.num_steps)):
            # print(f"------------------------------- Time step {t} -------------------------------")
            # reset_nominal = False
            reset_nominal = (t == 0)  # reset nominal trajectory at the first step
            action, mode, expanded_region = self.controller.solve(self.state, reset_nominal, verbose=False)
            # print(f"self.controller.reached_goal: {self.controller.reached_goal}, self.controller.terminate: {self.controller.terminate}")
            if self.controller.reached_goal or self.controller.terminate:
                # print("Goal reached, stopping simulation.")
                self.state_log[t:] = self.state  # fill remaining state log with current state
                self.control_log[t:] = 0.0  # fill remaining control log with zeros
                self.mode_log[t:] = mode  # fill remaining mode log with current mode
                self.expanded_region_log.extend([None] * (self.num_steps - t))  # fill remaining expanded region log with None
                break
            self.expanded_region_log.append(expanded_region)
            # print(f"Step {t}, State: {self.state}, Action: {action}, Mode: {mode}")
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            self.mode_log[t] = mode
            # self.state, opponent_feedbacks = self.controller.mppi_controller_fast.simulate_step(self.state, action)
            ## piece-wise constant intent example
            # if int(t/3)
            # self.controller.mppi_cfg.opponent_gain = ((self.num_steps - t)//3)*0.07 + 0.5 #(t//3)*0.07 + 0.5

            #piecewise constant intent example
            # if t < self.num_steps // 3:
            #     self.controller.mppi_cfg.opponent_gain = 0.5
            # elif t < 2 * self.num_steps // 3:
            #     self.controller.mppi_cfg.opponent_gain = 0.8
            # else:
            #     self.controller.mppi_cfg.opponent_gain = 1.1

            # if t < self.num_steps // 3:
            #     self.controller.mppi_cfg.opponent_gain = 0.5
            # else:
            #     self.controller.mppi_cfg.opponent_gain = 1.0
            self.controller.mppi_cfg.opponent_gain = 0.5
            # self.controller.mppi_controller_local.mppi_cfg.opponent_gain = self.controller.mppi_cfg.opponent_gain
            # self.controller.mppi_controller_fast.mppi_cfg.opponent_gain = self.controller.mppi_cfg.opponent_gain
            # print(f"True opponent control gain at step {t}: {self.controller.mppi_cfg.opponent_gain}")
            ##
            next_state, opponent_feedbacks = self.controller.mppi_controller_simulate_step.simulate_step(self.state, action)
            # print(f"Step {t}, State: {self.state}, Next State: {next_state}, Action: {action}, Mode: {mode}")
            self.state = next_state
            # print(f"Opponent feedbacks: {opponent_feedbacks[1]}")
            # Update control gain estimator with new observation
            self.control_gain_estimator.update_window(self.state, opponent_feedbacks[1])

            if t % 5 == 0:
                estimated_gain = self.control_gain_estimator.estimate_control_gain()
                # print(f"Estimated opponent control gain at step {t}: {estimated_gain}")
                if estimated_gain is not None:
                    if abs(estimated_gain - self.controller.mppi_cfg.opponent_gain) > 0.1:
                        self.controller.mppi_cfg.opponent_gain = estimated_gain
                        self.controller.recompute = True  # set flag to recompute verified set with new opponent gain
                # print(f"Updated opponent control gain to: {self.controller.mppi_cfg.opponent_gain}")


        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
            "mode_log": self.mode_log,
            "expanded_region_log": self.expanded_region_log,
        }
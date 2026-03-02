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

from local_verif_utils import get_beta5, beta, calibrate_V_vectorized, calibrate_V_scenario_local_vectorized, grow_regions_closest_point, grow_regions_closest_point_new, make_new_env, compute_min_scenarios_alex
from time import time

# set random seeds for reproducibility
seed = 12 #11 #0
np.random.seed(seed)
torch.manual_seed(seed)

DEFAULT_CTRL_CFG = DroneMPCConfig()

@dataclass
class DroneMPPIConfig:
    """Configuration container for the drone MPPI controller."""

    horizon: int = 15 #25 # Ebonye 1/27/2026
    dt: float = 0.1
    num_samples: int = 500 #1000
    temperature: float = 10.0 #1.0
    noise_sigma: Sequence[float] = (0.4, 0.4, 0.4)
    position_weight: float = 5.0
    velocity_weight: float = 1.0
    control_weight: float = 0.1
    safety_radius: float = 0.2
    safety_weight: float = 0.0
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
        value = self.interpolator(state)
        # print(f"Verified value function at state {state}: {value}")
        return value > 0.0

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

        # V_lp_scenario_vectorized_flat, _, _ = calibrate_V_vectorized(
        #     env,
        #     policy_function,
        #     tmp_states_reshaped,
        #     self.reachability_horizon,
        #     self.alphaC_scenario_list,
        #     self.alphaR_scenario_list,
        #     args,
        #     self.gamma
        # )
        # V_lp_scenario_vectorized = V_lp_scenario_vectorized_flat.reshape(H, W)

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
        self.mppi_cfg = mppi_cfg
        # self.mppi_cfg.horizon = 20  # increase horizon for better performance in the local MPPI mode
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
    
    def generate_straight_line_reference(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        num_points: int,
        target_speed: float,
        dt: float = 0.1,
    ) -> np.ndarray:
        """
        Create ego reference states [x, vx, y, vy, z, vz] along a straight line.
        Motion advances by target_speed * dt each step (constant-speed, dynamically consistent).
        After reaching the goal, continues forward in the same direction.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        start_xy = np.asarray(start_xy, dtype=np.float64)[[0, 2]]
        goal_xy = np.asarray(goal_xy, dtype=np.float64)[[0, 2]]

        direction = goal_xy - start_xy
        distance = np.linalg.norm(direction)

        if distance < 1e-8:
            unit_dir = np.array([0.0, 0.0])
        else:
            unit_dir = direction / distance

        step_dist = target_speed * dt

        # Initialize arrays
        xy = np.zeros((num_points, 2), dtype=np.float64)
        velocities = np.zeros((num_points, 2), dtype=np.float64)

        xy[0] = start_xy
        velocities[0] = target_speed * unit_dir

        # March forward at constant arc-length (same idea as quarter circle)
        for i in range(1, num_points):
            xy[i] = xy[i - 1] + unit_dir * step_dist
            velocities[i] = target_speed * unit_dir

        # Optional: if you want to stop at the goal instead of passing through,
        # uncomment this block:
        #
        # dist_along_path = np.linalg.norm(xy - start_xy, axis=1)
        # reached_goal = dist_along_path >= distance
        # if np.any(reached_goal):
        #     first_goal_idx = int(np.argmax(reached_goal))
        #     xy[first_goal_idx] = goal_xy
        #     velocities[first_goal_idx:] = 0.0
        #     for i in range(first_goal_idx + 1, num_points):
        #         xy[i] = goal_xy

        # Build full state [x, vx, y, vy, z, vz]
        reference = np.zeros((num_points, 6), dtype=np.float64)
        reference[:, 0] = xy[:, 0]
        reference[:, 2] = xy[:, 1]
        reference[:, 4] = 0.0  # constant altitude

        reference[:, 1] = velocities[:, 0]
        reference[:, 3] = velocities[:, 1]
        reference[:, 5] = 0.0

        return reference
    
    def generate_gate_pass_reference(
            self,
            current_state: np.ndarray,
            num_points: int,
            target_speed: float,
            gate_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
            dt: float = 0.1,
            lookahead_time: float = 0.2,
    ) -> np.ndarray:
        """
        Generate reference states to pass through the gate at a target speed.
        """
        # Define racing axis
        unit_dir = np.array([0.0, 1.0, 0.0])  # assuming gate is aligned with y-axis

        # Project current position onto racing axis relative to gate
        pos = current_state[[0, 2, 4]]
        dist_from_gate = np.dot(pos - gate_position, unit_dir)

        # Calculate 'anchor' distance along the racing axis to start the reference trajectory
        start_dist = dist_from_gate + (target_speed * lookahead_time)

        reference = np.zeros((num_points, 6), dtype=np.float64)

        # Populate the horizon
        for i in range(num_points):
            progress = start_dist + (target_speed * dt * i)
            ref_pos = gate_position + (progress * unit_dir)
            ref_vel = target_speed * unit_dir
            reference[i, 0] = ref_pos[0]
            reference[i, 1] = ref_vel[0]
            reference[i, 2] = ref_pos[1]
            reference[i, 3] = ref_vel[1]
            reference[i, 4] = ref_pos[2]
            reference[i, 5] = ref_vel[2]
        return reference
    
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
    
    def generate_straight_line_reference2(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        num_points: int,
        desired_speed: float = 1.5,
    ) -> np.ndarray:
        """
        Generate reference states for maintaining high speed along a straight line from start_state to goal_state.
        """
        state_dim = start_state.shape[0]
        per_agent_state_dim = self.mppi_cfg.per_agent_state_dim
        if state_dim % per_agent_state_dim != 0:
            raise ValueError("start_state has incompatible dimension with policy_function.")
        num_agents = state_dim // per_agent_state_dim
        reference = np.zeros((num_points, per_agent_state_dim), dtype=np.float64)

        direction = goal_state[[0, 2]] - start_state[[0, 2]]
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            unit_direction = np.array([0.0, 0.0])
        else:
            unit_direction = direction / direction_norm

        for t in range(num_points):
            position = start_state[[0, 2]] + (t / (num_points - 1)) * direction
            velocity = desired_speed * unit_direction
            # reference[t] = np.array([position[0], velocity[0], position[1], velocity[1], start_state[4], 0.0])  # keep z position and velocity zero
            reference[t] = np.array([position[0], velocity[0], position[1], velocity[1], 0.0, 0.0])  # keep z position and velocity zero
        
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
                self.recompute = False  # reset flag after recomputing verified set
        
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
                        max_attept_radius = 1.0, #0.5,
                        N_samples = n_samples,
                        tol=1e-2
                    )
                    self.expanded_region = expanded_region
                    x_center, y_center, r_safe = expanded_region
                    # print(f"state: {state} ")
                    if r_safe == 0:
                        if verbose:
                            print("Warning: There is no zero-level set for the value function, local growth failed. Trying to grow from target set instead.")
                        # plot entire value function for debugging
                        # import seaborn as sns
                        # import matplotlib.pyplot as plt
                        # x_interval = 5
                        # y_interval = 10
                        # fig, ax = plt.subplots(figsize=(8, 6))
                        # V_lp_flipped = np.flipud(self.verified_reachable_set.value_function)
                        # sns.heatmap(V_lp_flipped, annot=False, cmap=cm.coolwarm_r, alpha=0.9, ax=ax, cbar=True)
                        # x_ticks = np.arange(0, len(x), x_interval)
                        # y_ticks = np.arange(0, len(y), y_interval)
                        # contours_debug = ax.contour((X+0.9)*10, (Y+2.6)*10, V_lp_flipped, levels=[0], colors='black', alpha=0.3)
                        # ax.set_xticks(x_ticks)
                        # ax.set_yticks(y_ticks)

                        # ax.set_xticklabels(np.round(x[::x_interval], 2))
                        # ax.set_yticklabels(np.round(y[::-y_interval], 1))
                        # contours1 = ax.contour((X+0.9)*10, (Y+2.6)*10, V_lp_flipped, levels=[0], colors='black', linestyles='dashed')
                        # plt.savefig("debug_full_value_function.png")

                        # try to grow a region from the target set instead
                        # print(f"state before growing region: {state} ")
                        state_target = target_set(X, Y, state.copy())
                        # plot target set for debugging
                        # fig_target, ax_target = plt.subplots(figsize=(8, 6))
                        # target_contours = ax_target.contourf(X, Y, state_target, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
                        # plt.savefig("debug_target_set.png")
                        # import pdb; pdb.set_trace()

                        # print(f"state after growth: {state} ")
                        #put expanded region in figure with target set
                        # x_center, y_center, r_safe = expanded_region
                        # circle = plt.Circle((x_center, y_center), r_safe, color='blue', fill=False, linestyle='dashed', label='Expanded Region')
                        # ax_target.add_artist(circle)

                        # print(f"state before growth: {state} ")
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
                            state_target,
                            max_attept_radius = 1.0, #0.5,
                            N_samples = n_samples,
                            tol=1e-2,
                            target=True)
                        x_center_target, y_center_target, r_safe_target = expanded_region
                        self.expanded_region = expanded_region
                        # if r_safe_target == 0:
                        #     print("Warning: Local growth from target set also failed, no safe region found. Defaulting to MPPI towards target set that excludes velocity constraint.")
                        #     target_set_modified = target_set_last_resort(X, Y, state.copy())
                        #     self.target_set_modified = target_set_modified
                        #     # expanded_region = (x_center_target, y_center_target, r_safe_target)  # use the failed growth from target set as the "expanded region" for MPPI to try to reach towards


                    # x_center, y_center, r_safe = expanded_region
                    self.expanded_region = expanded_region
                

                    # import pdb; pdb.set_trace()
                    # check if current state is inside local growth set
                    # is_safe_local = np.linalg.norm(ego_xy - np.array([x_center, y_center])) <= r_safe
                    # self.is_safe_local = is_safe_local
                    # print(f"is_safe_local: {is_safe_local}")
                    self.recompute_local = False  # reset flag after recomputing local growth set
                    end_time = time()
                    if verbose:
                        print(f"Time taken to compute local growth set: {end_time - start_time:.2f} seconds")

                # print(f"state: {state} ")
                # print(f"expanded_region: {self.expanded_region} ")
                # print(f"recompute: {self.recompute}, recompute_local: {self.recompute_local} ")
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
                    # no local growth set found: use MPPI towards closest point in target set without velocity constraint
                    if verbose:
                        print(f"No local growth set found, using MPPI towards closest point in target set without velocity constraint.")
                    target_set_modified = target_set_last_resort(X, Y, state.copy())
                    # self.target_set_modified = target_set_modified
                    # find closest point in modified target set
                    target_points = np.column_stack((X.flatten(), Y.flatten()))
                    target_values = target_set_modified.flatten()
                    valid_indices = np.where(target_values > 0)[0]

                    ego_xy = state[[0, 2]]
                    diff = target_points[valid_indices] - ego_xy
                    distances = np.linalg.norm(diff, axis=1)
                    # print(f"state: {state}")
                    # print(f"len(distances): {len(distances)}")
                    # print(f"np.argmin(distances): {np.argmin(distances)}")
                    closest_xy = target_points[valid_indices[np.argmin(distances)]]
                    closest_point = np.array([closest_xy[0], closest_xy[1], 0.0])  # make z coordinate 0
                    
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
            # desired_position = np.array([x_star[0], state[2], state[4]])  # keep current y, z positions
            # # desired_velocity = np.array([0.0, desired_speed, 0.0])  # high speed along y-axis
            # desired_velocity = np.array([desired_speed, desired_speed, 0.0])  # high speed along x and y-axis
            # desired_state = np.array([desired_position[0], desired_velocity[0],
            #                  desired_position[1], desired_velocity[1],
            #                  desired_position[2], desired_velocity[2]])
            # reference = np.tile(desired_state, (self.mppi_cfg.horizon, 1))
            ###
            goal_state = x_star.copy()
            goal_state[2] = 0.5  # set goal y position to be just past the gate to encourage more aggressive behavior in passing through the gate
            # print(f"Goal state for high-speed lane maintain: {goal_state}")

            # reference = self.generate_straight_line_reference(
            #     state,
            #     goal_state,
            #     self.mppi_fast_cfg.horizon,
            #     desired_speed,
            # )
            # reference = self.generate_gate_pass_reference(
            #     state,
            #     self.mppi_fast_cfg.horizon,
            #     desired_speed,
            #     gate_position=np.array([0.0, 0.0, 0.0]),  # set gate position to be just past the actual gate to encourage more aggressive behavior in passing through the gate
            #     dt=self.mppi_fast_cfg.dt,
            #     lookahead_time=0.2,
            # )

            # reference = np.tile(goal_state, (self.mppi_fast_cfg.horizon, 1))

            # print(f"state: {state}")
            reference = self.generate_sliding_track_reference(
                state,
                self.mppi_fast_cfg.horizon,
                desired_speed,
                dt=self.mppi_fast_cfg.dt,
            )
            # import pdb; pdb.set_trace()
            # print(f"Generated sliding track reference for high-speed lane maintain MPPI: {reference}")
            # import pdb; pdb.set_trace()


            # print(f"Generated gate pass reference for high-speed lane maintain MPPI: {reference}")
            # import pdb; pdb.set_trace()
            # print(f"Generated straight line reference: {reference}")
            # self.expanded_region = None  # no expanded region in this mode
            ###``
            # ref_traj = np.tile(desired_position, (self.mppi_cfg.horizon, 1))
            # ref_velocities = np.tile(np.array([desired_speed, 0.0, 0.0]), (self.mppi_cfg.horizon, 1))
            # self.mppi_controller_fast._set_reference(ref_traj, ref_velocities)
            # action = self.mppi_controller_fast.solve(state, reset_nominal)
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
            # reference = self.generate_straight_line_reference(
            #     state,
            #     goal_state,
            #     self.mppi_fast_cfg.horizon,
            #     desired_speed,
            # )
            # print(f"state: {state}")
            # reference = self.generate_gate_pass_reference(
            #     state,
            #     self.mppi_fast_cfg.horizon,
            #     desired_speed,
            #     gate_position=np.array([0.0, 0.0, 0.0]),  # set gate position to be just past the actual gate to encourage more aggressive behavior in passing through the gate
            #     dt=self.mppi_fast_cfg.dt,
            #     lookahead_time=0.2,
            # )
            reference = self.generate_sliding_track_reference(
                state,
                self.mppi_fast_cfg.horizon,
                desired_speed,
                dt=self.mppi_fast_cfg.dt,
            )
            # import pdb; pdb.set_trace()
            # print(f"Generated gate pass reference for near gate high-speed MPPI: {reference}")
            # self.expanded_region = None  # no expanded region in this mode
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
            action, mode, expanded_region = self.controller.solve(self.state, self.control_log[t-1] if t > 0 else None, reset_nominal, verbose=True)
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

class DroneRaceMPPIBaselineSimulation:
    """Simulates an ego drone trying to overtake another drone using MPPI baseline without any switching or verified set."""

    def __init__(
            self,
            args,
            initial_state: np.ndarray,
            sim_cfg: DroneRaceConfig,
            mppi_cfg: DroneMPPIConfig,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.mppi_cfg = mppi_cfg
        self.mppi_cfg.horizon = 25  # increase horizon for better performance in the MPPI baseline
        self.mppi_cfg.safety_radius = 0.2
        self.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the MPPI baseline
        self.mppi_cfg.safety_weight = 0.0  # increase safety weight to encourage more conservative behavior in the MPPI baseline
        self.mppi_cfg.velocity_weight = 2.0  # increase velocity weight for high-speed lane maintain behavior in the MPPI baseline
        self.dt = self.mppi_cfg.dt
        self.num_steps = int(np.ceil(sim_cfg.duration / self.dt))
        # self.mppi_controller = DroneMPPIController(self.mppi_cfg)
        self.mppi_controller = DroneMPPIControllerFast(self.mppi_cfg)
        self.state = initial_state
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, self.mppi_cfg.per_agent_control_dim), dtype=np.float64)
        self.reference_traj = np.zeros((self.num_steps, self.mppi_cfg.per_agent_state_dim), dtype=np.float64)

    def generate_straight_line_reference2(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        num_points: int,
        desired_speed: float = 1.5,
    ) -> np.ndarray:
        """
        Generate reference states for maintaining high speed along a straight line from start_state to goal_state.
        """
        state_dim = start_state.shape[0]
        per_agent_state_dim = self.mppi_cfg.per_agent_state_dim
        if state_dim % per_agent_state_dim != 0:
            raise ValueError("start_state has incompatible dimension with policy_function.")
        num_agents = state_dim // per_agent_state_dim
        reference = np.zeros((num_points, per_agent_state_dim), dtype=np.float64)

        direction = goal_state[[0, 2]] - start_state[[0, 2]]
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            unit_direction = np.array([0.0, 0.0])
        else:
            unit_direction = direction / direction_norm

        for t in range(num_points):
            position = start_state[[0, 2]] + (t / (num_points - 1)) * direction
            velocity = desired_speed * unit_direction
            # reference[t] = np.array([position[0], velocity[0], position[1], velocity[1], start_state[4], 0.0])  # keep z position and velocity zero
            reference[t] = np.array([position[0], velocity[0], position[1], velocity[1], 0.0, 0.0])  # keep z position and velocity zero
        
        
        return reference
    
    def generate_straight_line_reference(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        num_points: int,
        target_speed: float,
        dt: float,
    ) -> np.ndarray:
        """
        Create ego reference states [x, vx, y, vy, z, vz] along a straight line.
        Motion advances by target_speed * dt each step (constant-speed, dynamically consistent).
        After reaching the goal, continues forward in the same direction.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        start_xy = np.asarray(start_xy, dtype=np.float64)[[0, 2]]
        goal_xy = np.asarray(goal_xy, dtype=np.float64)[[0, 2]]

        direction = goal_xy - start_xy
        distance = np.linalg.norm(direction)

        if distance < 1e-8:
            unit_dir = np.array([0.0, 0.0])
        else:
            unit_dir = direction / distance

        step_dist = target_speed * dt

        # Initialize arrays
        xy = np.zeros((num_points, 2), dtype=np.float64)
        velocities = np.zeros((num_points, 2), dtype=np.float64)

        xy[0] = start_xy
        velocities[0] = target_speed * unit_dir

        # March forward at constant arc-length (same idea as quarter circle)
        for i in range(1, num_points):
            xy[i] = xy[i - 1] + unit_dir * step_dist
            velocities[i] = target_speed * unit_dir

        # Optional: if you want to stop at the goal instead of passing through,
        # uncomment this block:
        #
        # dist_along_path = np.linalg.norm(xy - start_xy, axis=1)
        # reached_goal = dist_along_path >= distance
        # if np.any(reached_goal):
        #     first_goal_idx = int(np.argmax(reached_goal))
        #     xy[first_goal_idx] = goal_xy
        #     velocities[first_goal_idx:] = 0.0
        #     for i in range(first_goal_idx + 1, num_points):
        #         xy[i] = goal_xy

        # Build full state [x, vx, y, vy, z, vz]
        reference = np.zeros((num_points, 6), dtype=np.float64)
        reference[:, 0] = xy[:, 0]
        reference[:, 2] = xy[:, 1]
        reference[:, 4] = 0.0  # constant altitude

        reference[:, 1] = velocities[:, 0]
        reference[:, 3] = velocities[:, 1]
        reference[:, 5] = 0.0

        return reference
    
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


    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""
        goal_state = self.mppi_controller._x_star.copy()
        goal_state[2] = 0.75  # set goal y position to be just past the gate
        desired_speed = 0.7  # m/s
        # self.reference_tra`j = self.generate_straight_line_reference(
        #     self.state,
        #     goal_state,
        #     self.num_steps,
        #     desired_speed,
        #     self.dt,
        # )
        # print(f"Generated straight line reference trajectory for MPPI baseline: {self.reference_traj}")

        # self.reference_traj = self.generate_sliding_track_reference(
        #     self.state,
        #     self.num_steps,
        #     desired_speed,
        #     self.dt,
        # )
        # print(f"Generated sliding track reference trajectory for MPPI baseline: {self.reference_traj}")

        for t in range(self.num_steps):
        # for t in tqdm(range(self.num_steps)):
            # print(f"------------------------------- Time step {t} -------------------------------")
            # ref_start = min(t, self.reference_traj.shape[0]-1)
            # ref_segment = self.reference_traj[ref_start:self.mppi_cfg.horizon+ref_start]
            ref_segment = self.generate_sliding_track_reference(
                self.state,
                self.mppi_cfg.horizon,
                desired_speed,
                dt=self.dt,
            )
            reset_nominal = (t == 0)
            action, _ = self.mppi_controller.solve(self.state, ref_segment, reset_nominal)
            # print(f"Step {t}, State: {self.state}, Action: {action}")
            reached_goal = self.state[2] > 0.0 and np.abs(self.state[0]) <= 0.3
            if reached_goal:
                # print("Goal reached, stopping simulation.")
                self.state_log[t:] = self.state  # log current state at the time of reaching goal
                self.control_log[t:] = action[:3]  # log control input at the time of reaching goal
                break
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            next_state, _ = self.mppi_controller.simulate_step(self.state, action)
            # print(f"Current state: {self.state}, Action: {action}, Next State: {next_state}")
            # print(f"Current state: \n{self.state}\nAction: {action}\nNext State: {next_state}")
            self.state = next_state

        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
        }
    

class DroneRaceMPPIBaselineWarmstartLearnedPolicySimulation:
    """Simulates an ego drone trying to overtake another drone using MPPI baseline without any switching or verified set."""

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
        self.mppi_cfg = mppi_cfg
        self.mppi_cfg.horizon = 25  # increase horizon for better performance in the MPPI baseline
        self.mppi_cfg.safety_radius = 0.2
        self.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the MPPI baseline
        self.mppi_cfg.safety_weight = 5.0  # increase safety weight to encourage more conservative behavior in the MPPI baseline
        self.mppi_cfg.velocity_weight = 2.0  # increase velocity weight for high-speed lane maintain behavior in the MPPI baseline
        value_fn = (
            ReachabilityValueFunction.from_policy_path(
                str(reachability_value_path),
                device="cpu",
            )
            if reachability_value_path is not None
            else None
        )
        self.policy_function = value_fn.policy
        # self.policy_function = policy_function.policy
        self.dt = self.mppi_cfg.dt
        self.num_steps = int(np.ceil(sim_cfg.duration / self.dt))
        self.mppi_controller = DroneMPPIController(self.mppi_cfg)
        self.state = initial_state
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, self.mppi_cfg.per_agent_control_dim), dtype=np.float64)
        self.reference_traj = np.zeros((self.num_steps, self.mppi_cfg.per_agent_state_dim), dtype=np.float64)
        self.reference_actions = np.zeros((self.num_steps, self.mppi_cfg.per_agent_control_dim), dtype=np.float64)

    def find_a(self,state):
            tmp_obs = np.array(state).reshape(1,-1)
            tmp_batch = Batch(obs = tmp_obs, info = Batch())
            # tmp = self.policy_function.policy(tmp_batch, model = "actor_old").act
            # act = self.policy_function.policy.map_action(tmp).cpu().detach().numpy().flatten()
            tmp = self.policy_function(tmp_batch, model = "actor_old").act
            act = self.policy_function.map_action(tmp).cpu().detach().numpy().flatten()
            
            return act

    def generate_straight_line_reference(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        num_points: int,
        target_speed: float,
        dt: float = 0.1,
    ) -> np.ndarray:
        """
        Create ego reference states [x, vx, y, vy, z, vz] along a straight line.
        Motion advances by target_speed * dt each step (constant-speed, dynamically consistent).
        After reaching the goal, continues forward in the same direction.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        start_xy = np.asarray(start_xy, dtype=np.float64)[[0, 2]]
        goal_xy = np.asarray(goal_xy, dtype=np.float64)[[0, 2]]

        direction = goal_xy - start_xy
        distance = np.linalg.norm(direction)

        if distance < 1e-8:
            unit_dir = np.array([0.0, 0.0])
        else:
            unit_dir = direction / distance

        step_dist = target_speed * dt

        # Initialize arrays
        xy = np.zeros((num_points, 2), dtype=np.float64)
        velocities = np.zeros((num_points, 2), dtype=np.float64)

        xy[0] = start_xy
        velocities[0] = target_speed * unit_dir

        # March forward at constant arc-length (same idea as quarter circle)
        for i in range(1, num_points):
            xy[i] = xy[i - 1] + unit_dir * step_dist
            velocities[i] = target_speed * unit_dir

        # Optional: if you want to stop at the goal instead of passing through,
        # uncomment this block:
        #
        # dist_along_path = np.linalg.norm(xy - start_xy, axis=1)
        # reached_goal = dist_along_path >= distance
        # if np.any(reached_goal):
        #     first_goal_idx = int(np.argmax(reached_goal))
        #     xy[first_goal_idx] = goal_xy
        #     velocities[first_goal_idx:] = 0.0
        #     for i in range(first_goal_idx + 1, num_points):
        #         xy[i] = goal_xy

        # Build full state [x, vx, y, vy, z, vz]
        reference = np.zeros((num_points, 6), dtype=np.float64)
        reference[:, 0] = xy[:, 0]
        reference[:, 2] = xy[:, 1]
        reference[:, 4] = 0.0  # constant altitude

        reference[:, 1] = velocities[:, 0]
        reference[:, 3] = velocities[:, 1]
        reference[:, 5] = 0.0

        return reference
    
    
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
            
            if ref_y > 0.0: #and state[2] <= -0.1:
                ref_y = 0.0

            ref_vy = target_speed
            ref_vx = 0.0
            ref_vz = 0.0

            reference[i] = np.array([ref_x, ref_vx, ref_y, ref_vy, ref_z, ref_vz])
        
        return reference


    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""
        goal_state = self.mppi_controller._x_star.copy()
        goal_state[2] = 0.75  # set goal y position to be just past the gate
        desired_speed = 0.7  # m/s
        # self.reference_traj = self.generate_straight_line_reference(
        #     self.state,
        #     goal_state,
        #     self.num_steps,
        #     desired_speed,
        #     self.dt,
        # )
        _, self.reference_actions = self.generate_learned_policy_reference(
            self.policy_function,
            self.state,
            self.mppi_cfg.horizon,
        )
        self.mppi_controller.nominal_sequence = self.reference_actions.copy()  # warmstart MPPI with learned policy reference actions
        # print(f"Generated straight line reference trajectory for MPPI baseline: {self.reference_traj}")


        for t in range(self.num_steps):
        # for t in tqdm(range(self.num_steps)):
            # print(f"------------------------------- Time step {t} -------------------------------")
            # ref_start = min(t, self.reference_traj.shape[0]-1)
            # ref_segment = self.reference_traj[ref_start:self.mppi_cfg.horizon+ref_start]
            ref_segment = self.generate_sliding_track_reference(
                self.state,
                self.mppi_cfg.horizon,
                desired_speed,
                dt=self.dt,
            )
            # reset_nominal = False
            reset_nominal = (t == 0)
            action, _ = self.mppi_controller.solve(self.state, ref_segment, reset_nominal)
            # print(f"Step {t}, State: {self.state}, Action: {action}")
            reached_goal = self.state[2] > 0.0 and np.abs(self.state[0]) <= 0.3
            if reached_goal:
                # print("Goal reached, stopping simulation.")
                self.state_log[t:] = self.state  # log current state at the time of reaching goal
                self.control_log[t:] = action[:3]  # log control input at the time of reaching goal
                break
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            next_state, _ = self.mppi_controller.simulate_step(self.state, action)
            # print(f"Current state: {self.state}, Action: {action}, Next State: {next_state}")
            # print(f"Current state: \n{self.state}\nAction: {action}\nNext State: {next_state}")
            self.state = next_state

        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
        }
    
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
    
class SwitchingDroneControllerNoWarmstartwithLearnedPolicy:
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
        self.mppi_cfg = mppi_cfg
        self.mppi_fast_cfg = deepcopy(mppi_cfg)
        self.mppi_fast_cfg.velocity_weight = 2.0  # increase velocity weight for high-speed lane maintain mode
        self.mppi_fast_cfg.horizon = 20  # increase horizon for high-speed lane maintain mode to encourage more foresight in maintaining high speed
        # self.mppi_controller = DroneMPPIController(self.mppi_cfg)
        self.mppi_cfg.safety_radius = 0.2
        self.mppi_cfg.safety_weight = 5.0  # increase safety weight to encourage more conservative behavior in the local MPPI mode
        self.mppi_controller_local = DroneMPPIControllerLocalVerif(self.mppi_cfg) #, value_function=policy_function)
        self.mppi_controller_fast = DroneMPPIController(self.mppi_fast_cfg) #s, value_function=policy_function)
        self.mppi_controller_fast_near_gate = DroneMPPIController(self.mppi_fast_cfg)
        self.mppi_controller_simulate_step = DroneMPPIController(self.mppi_cfg) #, value_function=policy_function)
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
        return reference
    
    def generate_gate_pass_reference(
            self,
            current_state: np.ndarray,
            num_points: int,
            target_speed: float,
            gate_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
            dt: float = 0.1,
            lookahead_time: float = 0.2,
    ) -> np.ndarray:
        """
        Generate reference states to pass through the gate at a target speed.
        """
        # Define racing axis
        unit_dir = np.array([0.0, 1.0, 0.0])  # assuming gate is aligned with y-axis

        # Project current position onto racing axis relative to gate
        pos = current_state[[0, 2, 4]]
        dist_from_gate = np.dot(pos - gate_position, unit_dir)

        # Calculate 'anchor' distance along the racing axis to start the reference trajectory
        start_dist = dist_from_gate + (target_speed * lookahead_time)

        reference = np.zeros((num_points, 6), dtype=np.float64)

        # Populate the horizon
        for i in range(num_points):
            progress = start_dist + (target_speed * dt * i)
            ref_pos = gate_position + (progress * unit_dir)
            ref_vel = target_speed * unit_dir
            reference[i, 0] = ref_pos[0]
            reference[i, 1] = ref_vel[0]
            reference[i, 2] = ref_pos[1]
            reference[i, 3] = ref_vel[1]
            reference[i, 4] = ref_pos[2]
            reference[i, 5] = ref_vel[2]
        return reference
    
    def generate_straight_line_reference(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        num_points: int,
        target_speed: float,
        dt: float = 0.1,
    ) -> np.ndarray:
        """
        Create ego reference states [x, vx, y, vy, z, vz] along a straight line.
        Motion advances by target_speed * dt each step (constant-speed, dynamically consistent).
        After reaching the goal, continues forward in the same direction.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        start_xy = np.asarray(start_xy, dtype=np.float64)[[0, 2]]
        goal_xy = np.asarray(goal_xy, dtype=np.float64)[[0, 2]]

        direction = goal_xy - start_xy
        distance = np.linalg.norm(direction)

        if distance < 1e-8:
            unit_dir = np.array([0.0, 0.0])
        else:
            unit_dir = direction / distance

        step_dist = target_speed * dt

        # Initialize arrays
        xy = np.zeros((num_points, 2), dtype=np.float64)
        velocities = np.zeros((num_points, 2), dtype=np.float64)

        xy[0] = start_xy
        velocities[0] = target_speed * unit_dir

        # March forward at constant arc-length (same idea as quarter circle)
        for i in range(1, num_points):
            xy[i] = xy[i - 1] + unit_dir * step_dist
            velocities[i] = target_speed * unit_dir

        # Optional: if you want to stop at the goal instead of passing through,
        # uncomment this block:
        #
        # dist_along_path = np.linalg.norm(xy - start_xy, axis=1)
        # reached_goal = dist_along_path >= distance
        # if np.any(reached_goal):
        #     first_goal_idx = int(np.argmax(reached_goal))
        #     xy[first_goal_idx] = goal_xy
        #     velocities[first_goal_idx:] = 0.0
        #     for i in range(first_goal_idx + 1, num_points):
        #         xy[i] = goal_xy

        # Build full state [x, vx, y, vy, z, vz]
        reference = np.zeros((num_points, 6), dtype=np.float64)
        reference[:, 0] = xy[:, 0]
        reference[:, 2] = xy[:, 1]
        reference[:, 4] = 0.0  # constant altitude

        reference[:, 1] = velocities[:, 0]
        reference[:, 3] = velocities[:, 1]
        reference[:, 5] = 0.0

        return reference
    
    def generate_straight_line_reference2(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        num_points: int,
        desired_speed: float = 1.5,
    ) -> np.ndarray:
        """
        Generate reference states for maintaining high speed along a straight line from start_state to goal_state.
        """
        state_dim = start_state.shape[0]
        per_agent_state_dim = self.mppi_cfg.per_agent_state_dim
        if state_dim % per_agent_state_dim != 0:
            raise ValueError("start_state has incompatible dimension with policy_function.")
        num_agents = state_dim // per_agent_state_dim
        reference = np.zeros((num_points, per_agent_state_dim), dtype=np.float64)

        direction = goal_state[[0, 2]] - start_state[[0, 2]]
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            unit_direction = np.array([0.0, 0.0])
        else:
            unit_direction = direction / direction_norm

        for t in range(num_points):
            position = start_state[[0, 2]] + (t / (num_points - 1)) * direction
            velocity = desired_speed * unit_direction
            # reference[t] = np.array([position[0], velocity[0], position[1], velocity[1], start_state[4], 0.0])  # keep z position and velocity zero
            reference[t] = np.array([position[0], velocity[0], position[1], velocity[1], 0.0, 0.0])  # keep z position and velocity zero
        
        return reference
    
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
            
            if ref_y > 0.0: #and state[2] <= -0.1:
                ref_y = 0.0

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
        self.target_set_modified = None
        self.terminate = False

    def solve(
        self,
        state: np.ndarray,
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

        if is_in_target_set and not self.is_in_target_set:
            if verbose:
                print("Entered target set!")
            self.is_in_target_set = True

        if verbose:
            print(f"is_in_target_set: {is_in_target_set}")
            print(f"self.expanded_region: {self.expanded_region}")

        near_gate = np.linalg.norm(state[2]-0.0) <= 0.1 and np.abs(state[0]) <= 0.3
        # self.near_gate = near_gate

        if verbose:
            print(f"near_gate: {self.near_gate}")

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
                self.recompute = False  # reset flag after recomputing verified set
        
            is_safe = self.verified_reachable_set.is_inside(ego_xy)
            if verbose:
                print(f"is_safe: {is_safe}")
            # import pdb; pdb.set_trace()

            if is_safe:
                # Inside verified BRS (given new intent): use learned policy
                # action = self.find_a(state)
                x = np.arange(-0.9, 0.9, self.epsilon_x)
                y = np.arange(-2.6, 0, self.epsilon_x)
                X, Y = np.meshgrid(x, y)
                target_set_local = target_set_last_resort(X, Y, state.copy())
                target_points = np.column_stack((X.flatten(), Y.flatten()))
                target_values = target_set_local.flatten()
                valid_indices = np.where(target_values > 0)[0]
                if len(valid_indices) == 0:
                    if verbose:
                        print("Warning: Target set is empty, cannot grow local region. Using straight line reference to goal instead.")
                    closest_point = np.array([0.0, 0.0, state[4]])  # just use current x,y and keep velocity the same
                else:
                    # print(len(valid_indices))
                    ego_xy = state[[0, 2]]
                    diff = target_points[valid_indices] - ego_xy
                    distances = np.linalg.norm(diff, axis=1)
                    closest_xy = target_points[valid_indices[np.argmin(distances)]]
                    closest_point = np.array([closest_xy[0], closest_xy[1], state[4]])  # keep velocity the same
                self.mppi_controller_local.closest_safe_point = closest_point

                # reference = self.generate_straight_line_reference(
                #     state,
                #     closest_point,
                #     self.mppi_cfg.horizon,
                #     target_speed=0.9,  # moderate speed towards local growth set
                # )
                reference = self.generate_gate_pass_reference(
                    state,
                    self.mppi_cfg.horizon,
                    target_speed=0.9,
                    gate_position=closest_point,
                    lookahead_time=0.3,  # lookahead time to start reference trajectory before reaching the gate
                )
                action, _ = self.mppi_controller_local.solve(state, reference, reset_nominal)

                mode = 0
                # action = [action[0], action[1], action[2], 0.0, 0.0, 0.0]  # zero angular rates
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
                    expanded_region, _, _, _ = grow_regions_closest_point(
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
                        max_attept_radius = 1.0, #0.5,
                        N_samples = n_samples,
                        tol=1e-2
                    )
                    self.expanded_region = expanded_region
                    x_center, y_center, r_safe = expanded_region
                    # print(f"state: {state} ")
                    if r_safe == 0:
                        if verbose:
                            print("Warning: There is no zero-level set for the value function, local growth failed. Trying to grow from target set instead.")
                        # plot entire value function for debugging
                        # import seaborn as sns
                        # import matplotlib.pyplot as plt
                        # x_interval = 5
                        # y_interval = 10
                        # fig, ax = plt.subplots(figsize=(8, 6))
                        # V_lp_flipped = np.flipud(self.verified_reachable_set.value_function)
                        # sns.heatmap(V_lp_flipped, annot=False, cmap=cm.coolwarm_r, alpha=0.9, ax=ax, cbar=True)
                        # x_ticks = np.arange(0, len(x), x_interval)
                        # y_ticks = np.arange(0, len(y), y_interval)
                        # contours_debug = ax.contour((X+0.9)*10, (Y+2.6)*10, V_lp_flipped, levels=[0], colors='black', alpha=0.3)
                        # ax.set_xticks(x_ticks)
                        # ax.set_yticks(y_ticks)

                        # ax.set_xticklabels(np.round(x[::x_interval], 2))
                        # ax.set_yticklabels(np.round(y[::-y_interval], 1))
                        # contours1 = ax.contour((X+0.9)*10, (Y+2.6)*10, V_lp_flipped, levels=[0], colors='black', linestyles='dashed')
                        # plt.savefig("debug_full_value_function.png")

                        # try to grow a region from the target set instead
                        # print(f"state before growing region: {state} ")
                        # state_target = target_set(X, Y, state.copy())
                        state_target = target_set_last_resort(X, Y, state.copy())
                        # plot target set for debugging
                        # fig_target, ax_target = plt.subplots(figsize=(8, 6))
                        # target_contours = ax_target.contourf(X, Y, state_target, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
                        # plt.savefig("debug_target_set.png")
                        # import pdb; pdb.set_trace()

                        # print(f"state after growth: {state} ")
                        #put expanded region in figure with target set
                        # x_center, y_center, r_safe = expanded_region
                        # circle = plt.Circle((x_center, y_center), r_safe, color='blue', fill=False, linestyle='dashed', label='Expanded Region')
                        # ax_target.add_artist(circle)

                        # print(f"state before growth: {state} ")
                        expanded_region, _, _, _ = grow_regions_closest_point(
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
                            state_target,
                            max_attept_radius = 1.0, #0.5,
                            N_samples = n_samples,
                            tol=1e-2,
                            target=True)
                        x_center_target, y_center_target, r_safe_target = expanded_region
                        self.expanded_region = expanded_region
                        # if r_safe_target == 0:
                        #     print("Warning: Local growth from target set also failed, no safe region found. Defaulting to MPPI towards target set that excludes velocity constraint.")
                        #     target_set_modified = target_set_last_resort(X, Y, state.copy())
                        #     self.target_set_modified = target_set_modified
                        #     # expanded_region = (x_center_target, y_center_target, r_safe_target)  # use the failed growth from target set as the "expanded region" for MPPI to try to reach towards


                    # x_center, y_center, r_safe = expanded_region
                    self.expanded_region = expanded_region
                

                    # import pdb; pdb.set_trace()
                    # check if current state is inside local growth set
                    # is_safe_local = np.linalg.norm(ego_xy - np.array([x_center, y_center])) <= r_safe
                    # self.is_safe_local = is_safe_local
                    # print(f"is_safe_local: {is_safe_local}")
                    self.recompute_local = False  # reset flag after recomputing local growth set
                    end_time = time()
                    if verbose:
                        print(f"Time taken to compute local growth set: {end_time - start_time:.2f} seconds")

                # print(f"state: {state} ")
                # print(f"expanded_region: {self.expanded_region} ")
                # print(f"recompute: {self.recompute}, recompute_local: {self.recompute_local} ")
                x_center, y_center, r_safe = self.expanded_region
                is_safe_local = np.linalg.norm(ego_xy - np.array([x_center, y_center])) <= r_safe
                self.is_safe_local = is_safe_local
                if verbose:
                    print(f"is_safe_local: {is_safe_local}")
                # if is_safe_local:
                if self.is_safe_local: #or r_safe == 0:
                    # inside local growth set: use learned policy
                    # action = self.find_a(state)
                    x = np.arange(-0.9, 0.9, self.epsilon_x)
                    y = np.arange(-2.6, 0, self.epsilon_x)
                    X, Y = np.meshgrid(x, y)
                    # target_set_local = target_set(X, Y, state.copy())
                    target_set_local = target_set_last_resort(X, Y, state.copy())
                    target_points = np.column_stack((X.flatten(), Y.flatten()))
                    target_values = target_set_local.flatten()
                    valid_indices = np.where(target_values > 0)[0]
                    if len(valid_indices) == 0:
                        if verbose:
                            print("Warning: Target set is empty, cannot generate reference. Using straight line reference to goal instead.")
                        closest_point = np.array([0.0, 0.0, state[4]])  # just use current x,y and keep velocity the same
                    else:
                        ego_xy = state[[0, 2]]
                        diff = target_points[valid_indices] - ego_xy
                        distances = np.linalg.norm(diff, axis=1)
                        closest_xy = target_points[valid_indices[np.argmin(distances)]]
                        closest_point = np.array([closest_xy[0], closest_xy[1], state[4]])  # keep velocity the same
                    self.mppi_controller_local.closest_safe_point = closest_point

                    # reference = self.generate_straight_line_reference(
                    #     state,
                    #     closest_point,
                    #     self.mppi_cfg.horizon,
                    #     target_speed=0.5,  # moderate speed towards local growth set
                    # )
                    reference = self.generate_gate_pass_reference(
                        state,
                        self.mppi_cfg.horizon,
                        target_speed=0.5,
                        gate_position=closest_point,
                        lookahead_time=0.3,  # lookahead time to start reference trajectory before reaching the gate
                    )
                    action, _ = self.mppi_controller_local.solve(state, reference, reset_nominal)

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
                    if verbose:
                        print(f"Note safe local and r_safe >0, closest_safe_point: {closest_safe_point}")

                    self.mppi_controller_local.closest_safe_point = closest_safe_point
                    # reference = self.generate_learned_policy_reference(
                    #     self.policy_function,
                    #     state,
                    #     self.mppi_cfg.horizon
                    # )
                    # reference = self.generate_straight_line_reference(
                    #     state,
                    #     closest_safe_point,
                    #     self.mppi_cfg.horizon,
                    #     target_speed=0.5,  # moderate speed towards local growth set
                    # )
                    reference = self.generate_gate_pass_reference(
                        state,
                        self.mppi_cfg.horizon,
                        target_speed=0.5,
                        gate_position=closest_safe_point,
                        lookahead_time=0.3,  # lookahead time to start reference trajectory before reaching the gate
                    )
                    action, _ = self.mppi_controller_local.solve(state, reference, reset_nominal)
                    mode = 1
                    return action, mode, self.expanded_region
                
                elif not self.is_safe_local and r_safe == 0:
                    # no local growth set found: use MPPI towards closest point in target set without velocity constraint
                    if verbose:
                        print(f"No local growth set found, using MPPI towards closest point in target set without velocity constraint.")
                    target_set_modified = target_set_last_resort(X, Y, state.copy())
                    # self.target_set_modified = target_set_modified
                    # find closest point in modified target set
                    target_points = np.column_stack((X.flatten(), Y.flatten()))
                    target_values = target_set_modified.flatten()
                    valid_indices = np.where(target_values > 0)[0]

                    if len(valid_indices) == 0:
                        if verbose:
                            print("Warning: Modified target set is empty, cannot generate reference. Using straight line reference to goal instead.")
                        closest_point = np.array([0.0, 0.0, state[4]])  # just use current x,y and keep velocity the same
                    else:

                        ego_xy = state[[0, 2]]
                        diff = target_points[valid_indices] - ego_xy
                        distances = np.linalg.norm(diff, axis=1)
                        # print(f"state: {state}")
                        # print(f"len(distances): {len(distances)}")
                        # print(f"np.argmin(distances): {np.argmin(distances)}")
                        closest_xy = target_points[valid_indices[np.argmin(distances)]]
                        closest_point = np.array([closest_xy[0], closest_xy[1], 0.0])  # make z coordinate 0
                    
                    self.mppi_controller_local.closest_safe_point = closest_point
                    # reference = self.generate_learned_policy_reference(
                    #     self.policy_function,
                    #     state,
                    #     self.mppi_cfg.horizon
                    # )
                    # reference = self.generate_straight_line_reference(
                    #     state,
                    #     closest_point,
                    #     self.mppi_cfg.horizon,
                    #     target_speed=0.5,  # moderate speed towards modified target set
                    # )
                    reference = self.generate_gate_pass_reference(
                        state,
                        self.mppi_cfg.horizon,
                        target_speed=0.5,
                        gate_position=closest_point,
                        lookahead_time=0.3,  # lookahead time to start reference trajectory before reaching the gate
                    )
                    action, _ = self.mppi_controller_local.solve(state, reference, reset_nominal)
                    mode = 1
                    return action, mode, self.expanded_region
        
        # else:
        if (self.is_in_target_set and not self.reached_goal):
            # Safely ahead of opponent: maintain high speed along lane center using MPPI
            desired_speed = 0.7 #0.5 #0.7  # m/s
            # desired_position = np.array([0.0, state[1], state[2]])  # keep current y, z positions
            x_star = self.mppi_controller_fast._x_star
            # desired_position = np.array([x_star[0], state[2], state[4]])  # keep current y, z positions
            # # desired_velocity = np.array([0.0, desired_speed, 0.0])  # high speed along y-axis
            # desired_velocity = np.array([desired_speed, desired_speed, 0.0])  # high speed along x and y-axis
            # desired_state = np.array([desired_position[0], desired_velocity[0],
            #                  desired_position[1], desired_velocity[1],
            #                  desired_position[2], desired_velocity[2]])
            # reference = np.tile(desired_state, (self.mppi_cfg.horizon, 1))
            ###
            goal_state = x_star.copy()
            goal_state[2] = 0.75  # set goal y position to be just past the gate to encourage more aggressive behavior in passing through the gate
            # print(f"Goal state for high-speed lane maintain: {goal_state}")

            # reference = self.generate_straight_line_reference(
            #     state,
            #     goal_state,
            #     self.mppi_fast_cfg.horizon,
            #     desired_speed,
            # )

            reference = self.generate_gate_pass_reference(
                state,
                self.mppi_fast_cfg.horizon,
                target_speed=desired_speed,
                gate_position=goal_state[[0, 2, 4]],
                lookahead_time=0.3,  # lookahead time to start reference trajectory before reaching the gate
            )
            # reference = self.generate_sliding_track_reference(
            #     state,
            #     self.mppi_fast_cfg.horizon,
            #     target_speed=desired_speed,
            #     dt=self.mppi_fast_cfg.dt,
            # )
            # print(f"Generated straight line reference: {reference}")
            # self.expanded_region = None  # no expanded region in this mode
            ###``
            # ref_traj = np.tile(desired_position, (self.mppi_cfg.horizon, 1))
            # ref_velocities = np.tile(np.array([desired_speed, 0.0, 0.0]), (self.mppi_cfg.horizon, 1))
            # self.mppi_controller_fast._set_reference(ref_traj, ref_velocities)
            # action = self.mppi_controller_fast.solve(state, reset_nominal)
            action, _ = self.mppi_controller_fast.solve(state, reference, reset_nominal)
            mode = 2
            return action, mode, None

        elif self.near_gate:
            # near the gate but not yet passed it: use high-speed MPPI with a reference that goes through the gate to encourage aggressive behavior in passing through the gate
            desired_speed = 0.7 #0.5 #0.7  # m/s

            x_star = self.mppi_controller_fast_near_gate._x_star
            goal_state = x_star.copy()
            goal_state[2] = 0.75  # set goal y position to be just past the gate to encourage more aggressive behavior in passing through the gate
            if verbose:
                print(f"Goal state for near gate high-speed MPPI: {goal_state}")
            # reference = self.generate_straight_line_reference(
            #     state,
            #     goal_state,
            #     self.mppi_fast_cfg.horizon,
            #     desired_speed,
            # )
            reference = self.generate_gate_pass_reference(
                state,
                self.mppi_fast_cfg.horizon,
                target_speed=desired_speed,
                gate_position=goal_state[[0, 2, 4]],
                lookahead_time=0.3,  # lookahead time to start reference trajectory before reaching the gate
            )
            # reference = self.generate_sliding_track_reference(
            #     state,
            #     self.mppi_fast_near_gate_cfg.horizon,
            #     target_speed=desired_speed,
            #     dt=self.mppi_fast_near_gate_cfg.dt,
            # )
            # self.expanded_region = None  # no expanded region in this mode
            # print(f"Generated straight line reference for near gate high-speed MPPI: {reference}")
            action, _ = self.mppi_controller_fast_near_gate.solve(state, reference, reset_nominal)
            mode = 2
            return action, mode, None
        
class DroneRaceSimulationSwitchingNoLearnedPolicy:
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
        # self.controller = SwitchingDroneController(
        #     args=args,
        #     num_steps=self.num_steps,
        #     mppi_cfg=self.mppi_cfg,
        #     policy_function=value_fn,
        #     verified_reachable_set=offline_verified_set,            
        # )
        self.controller = SwitchingDroneControllerNoWarmstartwithLearnedPolicy(
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
            reset_nominal = t == 0  # reset nominal trajectory at the first step
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


class DroneRaceMPPIBaselinewithCBFSimulation:
    """Simulates an ego drone trying to overtake another drone using MPPI baseline with CBF safety filter without any switching or verified set."""

    def __init__(
            self,
            args,
            initial_state: np.ndarray,
            sim_cfg: DroneRaceConfig,
            mppi_cbf_cfg: MPPI_MPC_CBF_ControllerConfig,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.mppi_cfg = mppi_cbf_cfg
        self.mppi_cfg.mppi_cfg.horizon = 20  # increase horizon for better performance in the MPPI baseline with CBF
        self.mppi_cfg.mppi_cfg.safety_radius = 0.2
        self.mppi_cfg.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the MPPI baseline
        self.mppi_cfg.mppi_cfg.safety_weight = 0.0  # increase safety weight to encourage more conservative behavior in the MPPI baseline
        self.mppi_cfg.mppi_cfg.velocity_weight = 2.0  # increase velocity weight for high-speed lane maintain behavior in the MPPI baseline
        self.dt = self.mppi_cfg.mppi_cfg.dt
        self.num_steps = int(np.ceil(sim_cfg.duration / self.dt)) # + self.mppi_cfg.mppi_cfg.horizon  # add extra steps to allow MPPI-CBF controller to finish its horizon after reaching the goal
        self.controller = MPPI_MPC_CBF_Controller(self.mppi_cfg)
        self.state = initial_state.copy()
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, self.mppi_cfg.mppi_cfg.per_agent_control_dim), dtype=np.float64)
        self.reference_traj = np.zeros((self.num_steps, self.mppi_cfg.mppi_cfg.per_agent_state_dim), dtype=np.float64)

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


    

    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""
        goal_state = self.controller._x_star.copy()
        goal_state[2] = 0.2  # set goal y position to be just past the gate
        desired_speed = 0.7  # m/s
        # self.reference_traj = self.generate_straight_line_reference(
        #     self.state,
        #     goal_state,
        #     self.num_steps,
        #     desired_speed=desired_speed,
        # )
        # print(f"Generated straight line reference trajectory for MPPI baseline: {self.reference_traj}")

        # import pdb; pdb.set_trace()
        for t in range(self.num_steps):
        # for t in tqdm(range(self.num_steps)):

            # print(f"------------------------------- Time step {t} -------------------------------")
            # ref_start = min(t, self.reference_traj.shape[0]-1)
            # ref_segment = self.reference_traj[ref_start:self.mppi_cfg.mppi_cfg.horizon+ref_start]
            ref_segment = self.generate_sliding_track_reference(
                self.state,
                self.mppi_cfg.mppi_cfg.horizon,
                target_speed=desired_speed,
                dt=self.mppi_cfg.mppi_cfg.dt,
            )
            reset_nominal = False
            # action, _ = self.controller.solve(self.state, self.reference_traj, reset_nominal)
            action, info, nominal_action = self.controller.solve_total(self.state, ref_segment, reset_nominal)
            # print(f"CBF filter activated: {info['filtered']}")
            # print(f"Nominal action: {nominal_action}, Filtered action: {action}")
            # print(f"Step {t}, State: {self.state}, Action: {action}")
            reached_goal = self.state[2] > 0.0 and np.abs(self.state[0]) <= 0.3
            if reached_goal:
                # print("Goal reached, stopping simulation.")
                self.state_log[t:] = self.state  # log current state at the time of reaching goal
                self.control_log[t:] = action[:3]  # log control input at the time of reaching goal
                break
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            next_state, _ = self.controller.simulate_step(self.state, action)
            # print(f"Current state: {self.state}, Action: {action}, Next State: {next_state}")
            self.state = next_state

        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
        }

@dataclass
class MPCRaceConfig:
    """Scenario configuration for the MPC-CBF visualisation."""

    radius: float = 2.0
    target_speed: float = 0.8
    duration: float = 6.0
    save_path: pathlib.Path = pathlib.Path("experiment_script/data/mpc_cbf_traj.npz")
    opponent_y_offsets: Tuple[float, ...] = (-0.2, 0.2)

class MPCSimulation:
    """Simulate the CBF-constrained MPC controller in the drone racing setup."""

    def __init__(
        self,
        sim_cfg: DroneRaceConfig,
        ctrl_cfg: DroneMPCConfig,
        *,
        static_opponents: Tuple[int, ...] = (),
        debug: bool = False,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.ctrl_cfg = ctrl_cfg
        self.controller = DroneMPCCBFController(ctrl_cfg)
        self.static_opponents = {
            idx for idx in static_opponents if idx != self.controller.controlled_agent
        }
        self.debug = debug

        self.dt = ctrl_cfg.dt
        self.steps = int(np.ceil(sim_cfg.duration / self.dt))
        # add a few extra waypoints so the tail reference is available near the end
        self.reference = generate_quarter_circle_reference(
            sim_cfg.radius,
            self.steps + ctrl_cfg.horizon + 5,
            sim_cfg.target_speed,
            self.dt,
        )

        self.state = np.zeros(self.controller.state_dim, dtype=np.float64)
        self._initialise_state()

        self._K1 = np.array([3.1127], dtype=np.float64)
        self._K2 = np.array([9.1704, 16.8205], dtype=np.float64)
        self._x_star = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0], dtype=np.float64)
        self._opponent_gain = 1.0

        self.time_log = np.arange(self.steps, dtype=np.float64) * self.dt
        self.state_log = np.zeros((self.steps, self.controller.state_dim), dtype=np.float64)
        self.control_log = np.zeros((self.steps, self.controller.control_dim), dtype=np.float64)
        self.filter_log = np.zeros(self.steps, dtype=np.float64)
        self.distance_log = np.zeros(self.steps, dtype=np.float64)
        self.slack_log = np.zeros(self.steps, dtype=np.float64)

    @property
    def num_agents(self) -> int:
        return self.controller.num_agents

    def _initialise_state(self) -> None:
        """Place the controlled agent and two opponents along the start of the arc."""

        def state_on_arc(theta: float, z_offset: float = 0.0) -> np.ndarray:
            radius = self.sim_cfg.radius
            speed = self.sim_cfg.target_speed
            y = -radius * np.cos(theta)
            x = radius * np.sin(theta) - radius
            vy = speed * np.sin(theta)
            vx = speed * np.cos(theta)
            return np.array([x, vx, y, vy, z_offset, 0.0], dtype=np.float64)

        controlled_state = state_on_arc(0.0, 0.0)
        other_thetas = [0.12, 0.24]
        other_z = [-0.15, 0.15]
        other_states = [state_on_arc(theta, z) for theta, z in zip(other_thetas, other_z)]

        other_iter = iter(other_states)
        fallback = other_states[-1] if other_states else controlled_state
        offset_index = 0
        for idx in range(self.controller.num_agents):
            if idx == self.controller.controlled_agent:
                state_vec = controlled_state
            else:
                state_vec = next(other_iter, fallback).copy()
                if self.sim_cfg.opponent_y_offsets:
                    offset = self.sim_cfg.opponent_y_offsets[min(offset_index, len(self.sim_cfg.opponent_y_offsets) - 1)]
                    state_vec[2] += offset
                offset_index += 1
                if idx in self.static_opponents:
                    state_vec[1] = 0.0
                    state_vec[3] = 0.0
                    state_vec[5] = 0.0

            sl = slice(
                idx * self.controller.per_agent_state_dim,
                (idx + 1) * self.controller.per_agent_state_dim,
            )
            self.state[sl] = state_vec

    def _opponent_feedback(self, state: np.ndarray, agent_index: int) -> np.ndarray:
        block = state[
            agent_index * self.controller.per_agent_state_dim : (agent_index + 1) * self.controller.per_agent_state_dim
        ]
        ax = float(self._K2 @ np.array([self._x_star[0] - block[0], self._x_star[1] - block[1]]))
        ay = float(self._K1 @ np.array([self._x_star[3] - block[3]]))
        az = float(self._K2 @ np.array([self._x_star[4] - block[4], self._x_star[5] - block[5]]))
        return np.array([ax, ay, az], dtype=np.float64)

    def _simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Integrate the multi-agent double-integrator dynamics for one step."""

        next_state = np.array(state, dtype=np.float64, copy=True)
        dt = self.dt
        ctrl = np.clip(control, -self.ctrl_cfg.max_control, self.ctrl_cfg.max_control)
        per_agent = self.controller.per_agent_state_dim

        control_gain = self.ctrl_cfg.control_gain
        half_dt_sq = 0.5 * dt * dt

        for agent_idx in range(self.controller.num_agents):
            sl = slice(agent_idx * per_agent, (agent_idx + 1) * per_agent)
            block = next_state[sl]
            prev = state[sl]

            if agent_idx in self.static_opponents:
                next_state[sl] = prev
                continue

            if agent_idx == self.controller.controlled_agent:
                accel = control_gain * ctrl
            else:
                feedback = self._opponent_feedback(state, agent_idx)
                accel = self._opponent_gain * feedback

            block[0] = prev[0] + dt * prev[1] + half_dt_sq * accel[0]
            block[2] = prev[2] + dt * prev[3] + half_dt_sq * accel[1]
            block[4] = prev[4] + dt * prev[5] + half_dt_sq * accel[2]

            block[1] = prev[1] + dt * accel[0]
            block[3] = prev[3] + dt * accel[1]
            block[5] = prev[5] + dt * accel[2]

            next_state[sl] = block

        return next_state

    def _pairwise_distance(self, state: np.ndarray) -> float:
        per_agent = self.controller.per_agent_state_dim
        positions = []
        for idx in range(self.controller.num_agents):
            sl = slice(idx * per_agent, (idx + 1) * per_agent)
            block = state[sl]
            positions.append(np.array([block[0], block[2], block[4]], dtype=np.float64))
        positions = np.stack(positions)
        controlled = positions[self.controller.controlled_agent]
        others = np.delete(positions, self.controller.controlled_agent, axis=0)
        distances = np.linalg.norm(others - controlled, axis=1)
        return float(distances.min()) if distances.size else float("inf")

    def run(self) -> Dict[str, np.ndarray]:
        self.controller.reset_warm_start()
        num_ref = self.reference.shape[0]

        for t in range(self.steps):
            ref_start = min(t, num_ref - self.ctrl_cfg.horizon - 1)
            ref_segment = self.reference[ref_start : ref_start + self.ctrl_cfg.horizon]

            print(f"\n=== Step {t} ===")
            print(f"Current state pre-cbf: {self.state}")
            control, info = self.controller.solve(
                self.state,
                ref_segment,
                reset_nominal=False,
            )

            self.state_log[t] = self.state
            self.control_log[t] = control
            self.filter_log[t] = 1.0 if info.get("filtered") else 0.0
            self.distance_log[t] = self._pairwise_distance(self.state)
            slack_val = info.get("cbf_slack")
            self.slack_log[t] = float(slack_val) if slack_val is not None else np.nan
            if self.debug:
                violation = info.get("violation", float("nan"))
                print(
                    f"[step {t:03d}] min_dist={self.distance_log[t]:.3f} "
                    f"filtered={bool(info.get('filtered'))} status={info.get('filter_status')} "
                    f"violation={violation:.3f} slack={self.slack_log[t]:.3f}"
                )
                constraints = info.get("cbf_constraints", [])
                if constraints:
                    for c in constraints:
                        print(
                            f"    agent={c.get('agent_index')} h_cur={c.get('h_current', float('nan')):.3f} "
                            f"lower_bound={c.get('lower_bound', float('nan')):.3f} "
                            f"lin_nom={c.get('linearized_nominal', float('nan')):.3f} "
                            f"grad_norm={c.get('grad_norm', float('nan')):.3f}"
                        )
                else:
                    print("    no active CBF constraints")

            next_state = self._simulate_step(self.state, control)
            print(f"Next state post-cbf: {next_state}")
            self.state = next_state

        return {
            "time": self.time_log.copy(),
            "states": self.state_log.copy(),
            "controls": self.control_log.copy(),
            "filter_applied": self.filter_log.copy(),
            "min_distance": self.distance_log.copy(),
            "cbf_slack": self.slack_log.copy(),
        }

    def save(self, path: pathlib.Path, data: Dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)




def extract_xy(states: np.ndarray, agent_index: int, per_agent_state_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) position history for a given agent."""

    sl = slice(agent_index * per_agent_state_dim, (agent_index + 1) * per_agent_state_dim)
    agent_states = states[:, sl]
    return agent_states[:, 0], agent_states[:, 2]

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
            # print(f"X[{i},{j}]={X[i,j]}, Y[{i,j}]={Y[i,j]}, reward={reward[i,j]}")
            # print each term in the min function for debugging
            # print(f"tmp_point: {tmp_point}")
            # print(f"tmp_point[2] - tmp_point[8]: {tmp_point[2]} - {tmp_point[8]} = {tmp_point[2] - tmp_point[8]}")
            # # print(f"tmp_point[3] - tmp_point[9]: {tmp_point[3]} - {tmp_point[9]} = {tmp_point[3] - tmp_point[9]}")
            # print(f"tmp_point[0] - (-0.3): {tmp_point[0]} - (-0.3) = {tmp_point[0] - (-0.3)}")
            # print(f"0.3 - tmp_point[0]: 0.3 - {tmp_point[0]} = {0.3 - tmp_point[0]}")
            # print(f"tmp_point[4] - (-0.3): {tmp_point[4]} - (-0.3) = {tmp_point[4] - (-0.3)}")
            # print(f"0.3 - tmp_point[4]: 0.3 - {tmp_point[4]} = {0.3 - tmp_point[4]}")
    # import pdb; pdb.set_trace()
    return reward

        
def plot_trajectories(
    states: np.ndarray,
    # modes: np.ndarray,
    # expanded_regions: list[Tuple[float, float, float]],
    sim: DroneRaceSimulation,
    expanded_regions: list[Tuple[float, float, float]] = None,
    modes: np.ndarray = None,
    mppi_baseline_states: np.ndarray = None,
    save_path: pathlib.Path | None = None,
    gif_path: pathlib.Path | None = None,
    fps: int = 20,
) -> None:
    """Plot ego/opponent xy-trajectories and the quarter-circle reference."""
    print(f"expanded_regions len: {len(expanded_regions) if expanded_regions is not None else 'None'}")

    controller = sim.controller
    # per_agent_state_dim = controller.per_agent_state_dim
    per_agent_state_dim = controller.mppi_controller_fast.per_agent_state_dim

    fig, ax = plt.subplots(figsize=(6, 6))

    # vel magnitude per agent
    max_speeds = []
    max_speeds_mppi_baseline = []
    for idx in range(controller.mppi_controller_fast.num_agents):
        if idx == controller.mppi_controller_fast.controlled_agent:
            sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
            vel = states[:, sl][:, [1, 3, 5]]
            vel_mppi_baseline = mppi_baseline_states[:, sl][:, [1, 3, 5]] if mppi_baseline_states is not None else None
            speed = np.linalg.norm(vel, axis=1)
            speed_mppi_baseline = np.linalg.norm(vel_mppi_baseline, axis=1) if mppi_baseline_states is not None else None
            max_speeds.append(speed.max())
            max_speeds_mppi_baseline.append(speed_mppi_baseline.max()) if mppi_baseline_states is not None else None

    # # Reference quarter-circle
    # ref_xy = sim.reference[:, ::2][:, :2]
    # print(f"Ref xy shape: {ref_xy.shape}")
    # ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = []
    labels_mppi_baseline = []
    for idx in range(controller.mppi_controller_fast.num_agents):
        x, y = extract_xy(states, idx, per_agent_state_dim)
        x_mppi_baseline, y_mppi_baseline = extract_xy(mppi_baseline_states, idx, per_agent_state_dim) if mppi_baseline_states is not None else (None, None)
        lbl = "Ego agent" if idx == controller.mppi_controller_fast.controlled_agent else f"Agent {idx}"
        labels.append(ax.plot(x, y, color=colours[idx % len(colours)], label=lbl)[0])
        labels_mppi_baseline.append(ax.plot(x_mppi_baseline, y_mppi_baseline, color=colours[idx % len(colours)], linestyle="--", label=f"{lbl} (mppi baseline)")[0]) if mppi_baseline_states is not None else None

    
    # # Plotting target set 
    # x = np.arange(-0.9, 0.9, 0.01)
    # y = np.arange(-2.6, 0.0, 0.01)
    # X, Y = np.meshgrid(x, y)
    # for state in states:
    #     tmp_point = state.copy()
    #     Z = target_set(X, Y, tmp_point)
    #     ax.contourf(X, Y, Z, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
    #     ax.contour(X, Y, Z, levels=[1-1e-6, 1], colors=["green"], alpha=0.5, linewidths=1)

    

    ax.scatter(0.0, 0.0, marker="s", color="black", label="Gate (origin)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("MPPI Drone Trajectory")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-2.6, 0.2)
    ax.legend()

    print("Maximum speeds (m/s) per agent:")
    # for idx, ms in enumerate(max_speeds):
    if mppi_baseline_states is not None:
        for idx, (ms, ms_baseline) in enumerate(zip(max_speeds, max_speeds_mppi_baseline)):
            tag = "Switching Controller agent" if idx == controller.mppi_controller_fast.controlled_agent else f"Agent {idx}"
            tag_baseline = f"{tag} (MPPI baseline)"
            print(f"  {tag}: {ms:.3f}")
            print(f"  {tag_baseline}: {ms_baseline:.3f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    if gif_path is not None:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"title": "Drone MPPI", "artist": "visualize_mppi_traj"}
        fig_gif, ax_gif = plt.subplots(figsize=(6, 6))
        ax_gif.set_xlim(ax.get_xlim())
        ax_gif.set_ylim(ax.get_ylim())
        ax_gif.set_aspect("equal", adjustable="box")
        ax_gif.grid(True, linestyle="--", alpha=0.3)
        ax_gif.set_xlabel("x [m]")
        ax_gif.set_ylabel("y [m]")
        ax_gif.set_title("MPPI Drone Trajectory Animation")

        # ref_line, = ax_gif.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")
        gate_marker, = ax_gif.plot(0.0, 0.0, "s", color="black", label="Gate")

        trajectories = []
        velocities = []
        speed_components = []

        trajectories_baseline = []
        velocities_baseline = []
        speed_components_baseline = []
        modes_list = list(modes) if modes is not None else [None] * states.shape[0]
        for idx in range(controller.mppi_controller_fast.num_agents):
            # x, y = extract_xy(states, idx, per_agent_state_dim)
            # sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
            # agent_states = states[:, sl]
            # vx = agent_states[:, 1]
            # vy = agent_states[:, 3]
            # vz = agent_states[:, 5]
            # trajectories.append((x, y))
            # velocities.append((vx, vy))
            # speed_components.append((vx, vy, vz))
            
            if (idx == controller.mppi_controller_fast.controlled_agent and mppi_baseline_states is not None) or (mppi_baseline_states is None):
                x, y = extract_xy(states, idx, per_agent_state_dim)
                sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
                agent_states = states[:, sl]
                vx = agent_states[:, 1]
                vy = agent_states[:, 3]
                vz = agent_states[:, 5]
                trajectories.append((x, y))
                velocities.append((vx, vy))
                speed_components.append((vx, vy, vz))

            x, y = extract_xy(mppi_baseline_states, idx, per_agent_state_dim) if mppi_baseline_states is not None else (None, None)
            agent_states_baseline = mppi_baseline_states[:, sl] if mppi_baseline_states is not None else None
            vx_baseline = agent_states_baseline[:, 1] if agent_states_baseline is not None else None
            vy_baseline = agent_states_baseline[:, 3] if agent_states_baseline is not None else None
            vz_baseline = agent_states_baseline[:, 5] if agent_states_baseline is not None else None
            if mppi_baseline_states is not None:
                trajectories.append((x, y))
                velocities.append((vx_baseline, vy_baseline))
                speed_components.append((vx_baseline, vy_baseline, vz_baseline))
            # trajectories_baseline.append((x, y))
            # velocities_baseline.append((vx_baseline, vy_baseline))
            # speed_components_baseline.append((vx_baseline, vy_baseline, vz_baseline))
            
            # else:
            #     x = ()
            #     y = ()
            #     vx = ()
            #     vy = ()
            #     vz = ()
            #     trajectories_baseline.append((x, y))
            #     velocities_baseline.append((vx, vy))
            #     speed_components_baseline.append((vx, vy, vz))

        lines = []
        quivers = []
        speed_texts = []
        # contourf_list = []
        # contour_list = []
        for idx in range(controller.mppi_controller_fast.num_agents):
            colour = colours[idx % len(colours)]
            lbl = "Ego agent (Switching)" if idx == controller.mppi_controller_fast.controlled_agent else f"Agent {idx}"
            line, = ax_gif.plot([], [], color=colour, label=lbl)
            marker, = ax_gif.plot([], [], "o", color=colour)
            # contourf, = ax_gif.contourf([], [], [], levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
            # contour, = ax_gif.contour([], [], [], levels=[1-1e-6], colors=["green"], alpha=0.5, linewidths=1)
            quiver = ax_gif.quiver(
                [],
                [],
                [],
                [],
                color=colour,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.005,
            )
            lines.append((line, marker))
            quivers.append(quiver)
            # contourf_list.append(contourf)
            # contour_list.append(contour)
            text = ax_gif.text(
                0.0,
                0.0,
                "",
                color=colour,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
            )
            speed_texts.append(text)

            if idx == controller.mppi_controller_fast.controlled_agent and mppi_baseline_states is not None:
                colour_baseline = colour
                lbl_baseline = f"Ego agent (MPPI baseline)"
                line_baseline, = ax_gif.plot([], [], color=colour_baseline, linestyle="--", label=lbl_baseline)
                marker_baseline, = ax_gif.plot([], [], "o", color=colour_baseline, linestyle="--")
                quiver_baseline = ax_gif.quiver(
                    [],
                    [],
                    [],
                    [],
                    color=colour_baseline,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.005,
                    linestyle="--"
                )
                lines.append((line_baseline, marker_baseline))
                quivers.append(quiver_baseline)
                text_baseline = ax_gif.text(
                    0.0,
                    0.0,
                    "",
                    color=colour_baseline,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
                )
                speed_texts.append(text_baseline)

        # # plot target set in gif
        # x = np.arange(-0.9, 0.9, 0.01)
        # y = np.arange(-2.6, 0.0, 0.01)
        # X, Y = np.meshgrid(x, y)
        # for state in states:
        #     tmp_point = state.copy()
        #     Z = target_set(X, Y, tmp_point)
        #     ax_gif.contourf(X, Y, Z, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
        #     ax_gif.contour(X, Y, Z, levels=[1-1e-6, 1], colors=["green"], alpha=0.5, linewidths=1)

        ax_gif.legend()

        def init():
            artists = []
            # artists.extend([ref_line, gate_marker])
            for (line, marker), quiver, text in zip(lines, quivers, speed_texts):
                line.set_data([], [])
                marker.set_data([], [])
                quiver.set_offsets(np.empty((0, 2)))
                quiver.set_UVC([], [])
                text.set_text("")
                artists.extend([line, marker, quiver, text])
            return artists

        vel_arrow_scale = 0.5  # scales arrow length for visual clarity

        def update(frame: int):
            nonlocal contour, contourf
            artists = []
            x = np.arange(-0.9, 0.9, 0.01)
            y = np.arange(-2.6, 0.0, 0.01)
            X, Y = np.meshgrid(x, y)
            # for (
            #     (line, marker),
            #     quiver,
            #     text,
            #     (x_hist, y_hist),
            #     (vx_hist, vy_hist),
            #     (vx_comp, vy_comp, vz_comp),
            # ) in zip(
            #     lines, quivers, speed_texts, trajectories, velocities, speed_components
            # ):
            # print(f"Frame {frame}/{states.shape[0]}")
            # print(f"len(lines): {len(lines)}, len(quivers): {len(quivers)}, len(speed_texts): {len(speed_texts)}, len(trajectories): {len(trajectories)}, len(velocities): {len(velocities)}, len(speed_components): {len(speed_components)}")
            # print(f"len(trajectories_baseline): {len(trajectories_baseline)}, len(velocities_baseline): {len(velocities_baseline)}, len(speed_components_baseline): {len(speed_components_baseline)}")

            for i, (
                (line, marker),
                quiver,
                text,
                (x_hist, y_hist),
                (vx_hist, vy_hist),
                (vx_comp, vy_comp, vz_comp),
                # (x_hist_baseline, y_hist_baseline),
                # (vx_hist_baseline, vy_hist_baseline),
                # (vx_comp_baseline, vy_comp_baseline, vz_comp_baseline),
            ) in enumerate(zip(
                lines, quivers, speed_texts, trajectories, velocities, speed_components, #trajectories_baseline, velocities_baseline, speed_components_baseline 
            )):
                line.set_data(x_hist[: frame + 1], y_hist[: frame + 1])
                marker.set_data([x_hist[frame]], [y_hist[frame]])
                # include expanded region circle if not None for each frame
                # print(f"Agent {i}")
                if i == 0:  # only plot for ego agent
                    expanded_region = expanded_regions[frame] if expanded_regions is not None else None
                    if expanded_region is not None:
                        x_center, y_center, r_safe = expanded_region
                        circle = plt.Circle((x_center, y_center), r_safe, color='blue', fill=False, linestyle='dashed', label='Expanded Region')
                        # remove previous circle if exists
                        existing_circles = [artist for artist in ax_gif.patches if isinstance(artist, plt.Circle)]
                        for ec in existing_circles:
                            ec.remove()
                        # ax_gif.add_artist(circle)
                        ax_gif.add_patch(circle)
                    else:
                        # remove previous circle if exists
                        existing_circles = [artist for artist in ax_gif.patches if isinstance(artist, plt.Circle)]
                        for ec in existing_circles:
                            ec.remove()

                # update target set contours
                # tmp_point = states[frame].copy()
                tmp_point = mppi_baseline_states[frame].copy() if mppi_baseline_states is not None else states[frame].copy()
                Z = target_set(X, Y, tmp_point)
                if contourf is not None:
                    for c in contourf.collections:
                        c.remove()
                if contour is not None:
                    for c in contour.collections:
                        c.remove()
                # for c in contourf.collections:
                #     c.remove()
                # for c in contour.collections:
                #     c.remove()
                contourf = ax_gif.contourf(X, Y, Z, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
                contour = ax_gif.contour(X, Y, Z, levels=[1-1e-6, 1], colors=["green"], alpha=0.5, linewidths=1)



                vx = vx_hist[frame] * vel_arrow_scale
                vy = vy_hist[frame] * vel_arrow_scale
                quiver.set_offsets(np.array([[x_hist[frame], y_hist[frame]]]))
                quiver.set_UVC(np.array([vx]), np.array([vy]))
                speed = np.sqrt(
                    vx_comp[frame] ** 2 + vy_comp[frame] ** 2 + vz_comp[frame] ** 2
                )
                text.set_position((x_hist[frame] + 0.05, y_hist[frame] + 0.05))
                # text.set_text(f"{speed:.2f} m/s")
                mode = modes_list[frame] if modes_list is not None else None
                text.set_text(f"Mode: {mode}, Speed: {speed:.2f} m/s") if i == 0 else text.set_text(f"Speed: {speed:.2f} m/s")
                artists.extend([line, marker, quiver, text])
            
            # for i, (
            #     (line, marker),
            #     quiver,
            #     text,
            #     (x_hist_baseline, y_hist_baseline),
            #     (vx_hist_baseline, vy_hist_baseline),
            #     (vx_comp_baseline, vy_comp_baseline, vz_comp_baseline),
            # ) in enumerate(zip(
            #     lines, quivers, speed_texts, trajectories_baseline, velocities_baseline, speed_components_baseline
            # )):
            #     if len(x_hist_baseline) > 0:
            #         line.set_data(x_hist_baseline[: frame + 1], y_hist_baseline[: frame + 1])
            #         marker.set_data([x_hist_baseline[frame]], [y_hist_baseline[frame]])
            #         vx = vx_hist_baseline[frame] * vel_arrow_scale
            #         vy = vy_hist_baseline[frame] * vel_arrow_scale
            #         quiver.set_offsets(np.array([[x_hist_baseline[frame], y_hist_baseline[frame]]]))
            #         quiver.set_UVC(np.array([vx]), np.array([vy]))
            #         speed = np.sqrt(
            #             vx_comp_baseline[frame] ** 2 + vy_comp_baseline[frame] ** 2 + vz_comp_baseline[frame] ** 2
            #         )
            #         text.set_position((x_hist_baseline[frame] + 0.05, y_hist_baseline[frame] + 0.05))
            #         text.set_text(f"Speed: {speed:.2f} m/s")
            #         artists.extend([line, marker, quiver, text])
            
            
            return artists

        from matplotlib import animation

        blit_flag = gif_path is None
        contour = None
        contourf = None
        anim = animation.FuncAnimation(
            fig_gif,
            update,
            init_func=init,
            frames=states.shape[0],
            interval=1000 / fps,
            blit=blit_flag,
        )
        writer = animation.PillowWriter(fps=fps, metadata=metadata)
        anim.save(gif_path, writer=writer, dpi=100)
        plt.close(fig_gif)

def plot_trajectories2(
    states: np.ndarray,
    # modes: np.ndarray,
    # expanded_regions: list[Tuple[float, float, float]],
    sim: DroneRaceSimulation,
    expanded_regions: list[Tuple[float, float, float]] = None,
    modes: np.ndarray = None,
    mppi_baseline_states: np.ndarray = None,
    save_path: pathlib.Path | None = None,
    gif_path: pathlib.Path | None = None,
    fps: int = 20,
) -> None:
    """Plot ego/opponent xy-trajectories and the quarter-circle reference."""
    print(f"expanded_regions len: {len(expanded_regions) if expanded_regions is not None else 'None'}")

    controller = sim.controller
    per_agent_state_dim = controller.per_agent_state_dim
    # per_agent_state_dim = controller.mppi_controller_fast.per_agent_state_dim

    fig, ax = plt.subplots(figsize=(6, 6))

    # vel magnitude per agent
    max_speeds = []
    max_speeds_mppi_baseline = []
    for idx in range(controller.num_agents):
        if idx == controller.controlled_agent:
            sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
            vel = states[:, sl][:, [1, 3, 5]]
            vel_mppi_baseline = mppi_baseline_states[:, sl][:, [1, 3, 5]] if mppi_baseline_states is not None else None
            speed = np.linalg.norm(vel, axis=1)
            speed_mppi_baseline = np.linalg.norm(vel_mppi_baseline, axis=1) if mppi_baseline_states is not None else None
            max_speeds.append(speed.max())
            max_speeds_mppi_baseline.append(speed_mppi_baseline.max()) if mppi_baseline_states is not None else None

    # # Reference quarter-circle
    # ref_xy = sim.reference[:, ::2][:, :2]
    # print(f"Ref xy shape: {ref_xy.shape}")
    # ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = []
    labels_mppi_baseline = []
    for idx in range(controller.num_agents):
        x, y = extract_xy(states, idx, per_agent_state_dim)
        x_mppi_baseline, y_mppi_baseline = extract_xy(mppi_baseline_states, idx, per_agent_state_dim) if mppi_baseline_states is not None else (None, None)
        lbl = "Ego agent" if idx == controller.controlled_agent else f"Agent {idx}"
        labels.append(ax.plot(x, y, color=colours[idx % len(colours)], label=lbl)[0])
        labels_mppi_baseline.append(ax.plot(x_mppi_baseline, y_mppi_baseline, color=colours[idx % len(colours)], linestyle="--", label=f"{lbl} (mppi baseline)")[0]) if mppi_baseline_states is not None else None

    
    # # Plotting target set 
    # x = np.arange(-0.9, 0.9, 0.01)
    # y = np.arange(-2.6, 0.0, 0.01)
    # X, Y = np.meshgrid(x, y)
    # for state in states:
    #     tmp_point = state.copy()
    #     Z = target_set(X, Y, tmp_point)
    #     ax.contourf(X, Y, Z, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
    #     ax.contour(X, Y, Z, levels=[1-1e-6, 1], colors=["green"], alpha=0.5, linewidths=1)

    

    ax.scatter(0.0, 0.0, marker="s", color="black", label="Gate (origin)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("MPPI Drone Trajectory")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-2.6, 0.2)
    ax.legend()

    print("Maximum speeds (m/s) per agent:")
    # for idx, ms in enumerate(max_speeds):
    if mppi_baseline_states is not None:
        for idx, (ms, ms_baseline) in enumerate(zip(max_speeds, max_speeds_mppi_baseline)):
            tag = "Switching Controller agent" if idx == controller.controlled_agent else f"Agent {idx}"
            tag_baseline = f"{tag} (MPPI baseline)"
            print(f"  {tag}: {ms:.3f}")
            print(f"  {tag_baseline}: {ms_baseline:.3f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    if gif_path is not None:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"title": "Drone MPPI", "artist": "visualize_mppi_traj"}
        fig_gif, ax_gif = plt.subplots(figsize=(6, 6))
        ax_gif.set_xlim(ax.get_xlim())
        ax_gif.set_ylim(ax.get_ylim())
        ax_gif.set_aspect("equal", adjustable="box")
        ax_gif.grid(True, linestyle="--", alpha=0.3)
        ax_gif.set_xlabel("x [m]")
        ax_gif.set_ylabel("y [m]")
        ax_gif.set_title("MPPI Drone Trajectory Animation")

        # ref_line, = ax_gif.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")
        gate_marker, = ax_gif.plot(0.0, 0.0, "s", color="black", label="Gate")

        trajectories = []
        velocities = []
        speed_components = []

        trajectories_baseline = []
        velocities_baseline = []
        speed_components_baseline = []
        modes_list = list(modes) if modes is not None else [None] * states.shape[0]
        for idx in range(controller.num_agents):
            # x, y = extract_xy(states, idx, per_agent_state_dim)
            # sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
            # agent_states = states[:, sl]
            # vx = agent_states[:, 1]
            # vy = agent_states[:, 3]
            # vz = agent_states[:, 5]
            # trajectories.append((x, y))
            # velocities.append((vx, vy))
            # speed_components.append((vx, vy, vz))
            
            if (idx == controller.controlled_agent and mppi_baseline_states is not None) or (mppi_baseline_states is None):
                x, y = extract_xy(states, idx, per_agent_state_dim)
                sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
                agent_states = states[:, sl]
                vx = agent_states[:, 1]
                vy = agent_states[:, 3]
                vz = agent_states[:, 5]
                trajectories.append((x, y))
                velocities.append((vx, vy))
                speed_components.append((vx, vy, vz))

            x, y = extract_xy(mppi_baseline_states, idx, per_agent_state_dim) if mppi_baseline_states is not None else (None, None)
            agent_states_baseline = mppi_baseline_states[:, sl] if mppi_baseline_states is not None else None
            vx_baseline = agent_states_baseline[:, 1] if agent_states_baseline is not None else None
            vy_baseline = agent_states_baseline[:, 3] if agent_states_baseline is not None else None
            vz_baseline = agent_states_baseline[:, 5] if agent_states_baseline is not None else None
            if mppi_baseline_states is not None:
                trajectories.append((x, y))
                velocities.append((vx_baseline, vy_baseline))
                speed_components.append((vx_baseline, vy_baseline, vz_baseline))
            # trajectories_baseline.append((x, y))
            # velocities_baseline.append((vx_baseline, vy_baseline))
            # speed_components_baseline.append((vx_baseline, vy_baseline, vz_baseline))
            
            # else:
            #     x = ()
            #     y = ()
            #     vx = ()
            #     vy = ()
            #     vz = ()
            #     trajectories_baseline.append((x, y))
            #     velocities_baseline.append((vx, vy))
            #     speed_components_baseline.append((vx, vy, vz))

        lines = []
        quivers = []
        speed_texts = []
        # contourf_list = []
        # contour_list = []
        for idx in range(controller.num_agents):
            colour = colours[idx % len(colours)]
            lbl = "Ego agent (Switching)" if idx == controller.controlled_agent else f"Agent {idx}"
            line, = ax_gif.plot([], [], color=colour, label=lbl)
            marker, = ax_gif.plot([], [], "o", color=colour)
            # contourf, = ax_gif.contourf([], [], [], levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
            # contour, = ax_gif.contour([], [], [], levels=[1-1e-6], colors=["green"], alpha=0.5, linewidths=1)
            quiver = ax_gif.quiver(
                [],
                [],
                [],
                [],
                color=colour,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.005,
            )
            lines.append((line, marker))
            quivers.append(quiver)
            # contourf_list.append(contourf)
            # contour_list.append(contour)
            text = ax_gif.text(
                0.0,
                0.0,
                "",
                color=colour,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
            )
            speed_texts.append(text)

            if idx == controller.controlled_agent and mppi_baseline_states is not None:
                colour_baseline = colour
                lbl_baseline = f"Ego agent (MPPI baseline)"
                line_baseline, = ax_gif.plot([], [], color=colour_baseline, linestyle="--", label=lbl_baseline)
                marker_baseline, = ax_gif.plot([], [], "o", color=colour_baseline, linestyle="--")
                quiver_baseline = ax_gif.quiver(
                    [],
                    [],
                    [],
                    [],
                    color=colour_baseline,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.005,
                    linestyle="--"
                )
                lines.append((line_baseline, marker_baseline))
                quivers.append(quiver_baseline)
                text_baseline = ax_gif.text(
                    0.0,
                    0.0,
                    "",
                    color=colour_baseline,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
                )
                speed_texts.append(text_baseline)

        # # plot target set in gif
        # x = np.arange(-0.9, 0.9, 0.01)
        # y = np.arange(-2.6, 0.0, 0.01)
        # X, Y = np.meshgrid(x, y)
        # for state in states:
        #     tmp_point = state.copy()
        #     Z = target_set(X, Y, tmp_point)
        #     ax_gif.contourf(X, Y, Z, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
        #     ax_gif.contour(X, Y, Z, levels=[1-1e-6, 1], colors=["green"], alpha=0.5, linewidths=1)

        ax_gif.legend()

        def init():
            artists = []
            # artists.extend([ref_line, gate_marker])
            for (line, marker), quiver, text in zip(lines, quivers, speed_texts):
                line.set_data([], [])
                marker.set_data([], [])
                quiver.set_offsets(np.empty((0, 2)))
                quiver.set_UVC([], [])
                text.set_text("")
                artists.extend([line, marker, quiver, text])
            return artists

        vel_arrow_scale = 0.5  # scales arrow length for visual clarity

        def update(frame: int):
            nonlocal contour, contourf
            artists = []
            x = np.arange(-0.9, 0.9, 0.01)
            y = np.arange(-2.6, 0.0, 0.01)
            X, Y = np.meshgrid(x, y)
            # for (
            #     (line, marker),
            #     quiver,
            #     text,
            #     (x_hist, y_hist),
            #     (vx_hist, vy_hist),
            #     (vx_comp, vy_comp, vz_comp),
            # ) in zip(
            #     lines, quivers, speed_texts, trajectories, velocities, speed_components
            # ):
            # print(f"Frame {frame}/{states.shape[0]}")
            # print(f"len(lines): {len(lines)}, len(quivers): {len(quivers)}, len(speed_texts): {len(speed_texts)}, len(trajectories): {len(trajectories)}, len(velocities): {len(velocities)}, len(speed_components): {len(speed_components)}")
            # print(f"len(trajectories_baseline): {len(trajectories_baseline)}, len(velocities_baseline): {len(velocities_baseline)}, len(speed_components_baseline): {len(speed_components_baseline)}")

            for i, (
                (line, marker),
                quiver,
                text,
                (x_hist, y_hist),
                (vx_hist, vy_hist),
                (vx_comp, vy_comp, vz_comp),
                # (x_hist_baseline, y_hist_baseline),
                # (vx_hist_baseline, vy_hist_baseline),
                # (vx_comp_baseline, vy_comp_baseline, vz_comp_baseline),
            ) in enumerate(zip(
                lines, quivers, speed_texts, trajectories, velocities, speed_components, #trajectories_baseline, velocities_baseline, speed_components_baseline 
            )):
                line.set_data(x_hist[: frame + 1], y_hist[: frame + 1])
                marker.set_data([x_hist[frame]], [y_hist[frame]])
                # include expanded region circle if not None for each frame
                # print(f"Agent {i}")
                if i == 0:  # only plot for ego agent
                    expanded_region = expanded_regions[frame] if expanded_regions is not None else None
                    if expanded_region is not None:
                        x_center, y_center, r_safe = expanded_region
                        circle = plt.Circle((x_center, y_center), r_safe, color='blue', fill=False, linestyle='dashed', label='Expanded Region')
                        # remove previous circle if exists
                        existing_circles = [artist for artist in ax_gif.patches if isinstance(artist, plt.Circle)]
                        for ec in existing_circles:
                            ec.remove()
                        # ax_gif.add_artist(circle)
                        ax_gif.add_patch(circle)
                    else:
                        # remove previous circle if exists
                        existing_circles = [artist for artist in ax_gif.patches if isinstance(artist, plt.Circle)]
                        for ec in existing_circles:
                            ec.remove()

                # update target set contours
                # tmp_point = states[frame].copy()
                tmp_point = mppi_baseline_states[frame].copy() if mppi_baseline_states is not None else states[frame].copy()
                Z = target_set(X, Y, tmp_point)
                if contourf is not None:
                    for c in contourf.collections:
                        c.remove()
                if contour is not None:
                    for c in contour.collections:
                        c.remove()
                # for c in contourf.collections:
                #     c.remove()
                # for c in contour.collections:
                #     c.remove()
                contourf = ax_gif.contourf(X, Y, Z, levels=[1-1e-6, 1], colors=["lightgreen"], alpha=0.3)
                contour = ax_gif.contour(X, Y, Z, levels=[1-1e-6, 1], colors=["green"], alpha=0.5, linewidths=1)



                vx = vx_hist[frame] * vel_arrow_scale
                vy = vy_hist[frame] * vel_arrow_scale
                quiver.set_offsets(np.array([[x_hist[frame], y_hist[frame]]]))
                quiver.set_UVC(np.array([vx]), np.array([vy]))
                speed = np.sqrt(
                    vx_comp[frame] ** 2 + vy_comp[frame] ** 2 + vz_comp[frame] ** 2
                )
                text.set_position((x_hist[frame] + 0.05, y_hist[frame] + 0.05))
                # text.set_text(f"{speed:.2f} m/s")
                mode = modes_list[frame] if modes_list is not None else None
                text.set_text(f"Mode: {mode}, Speed: {speed:.2f} m/s") if i == 0 else text.set_text(f"Speed: {speed:.2f} m/s")
                artists.extend([line, marker, quiver, text])
            
            # for i, (
            #     (line, marker),
            #     quiver,
            #     text,
            #     (x_hist_baseline, y_hist_baseline),
            #     (vx_hist_baseline, vy_hist_baseline),
            #     (vx_comp_baseline, vy_comp_baseline, vz_comp_baseline),
            # ) in enumerate(zip(
            #     lines, quivers, speed_texts, trajectories_baseline, velocities_baseline, speed_components_baseline
            # )):
            #     if len(x_hist_baseline) > 0:
            #         line.set_data(x_hist_baseline[: frame + 1], y_hist_baseline[: frame + 1])
            #         marker.set_data([x_hist_baseline[frame]], [y_hist_baseline[frame]])
            #         vx = vx_hist_baseline[frame] * vel_arrow_scale
            #         vy = vy_hist_baseline[frame] * vel_arrow_scale
            #         quiver.set_offsets(np.array([[x_hist_baseline[frame], y_hist_baseline[frame]]]))
            #         quiver.set_UVC(np.array([vx]), np.array([vy]))
            #         speed = np.sqrt(
            #             vx_comp_baseline[frame] ** 2 + vy_comp_baseline[frame] ** 2 + vz_comp_baseline[frame] ** 2
            #         )
            #         text.set_position((x_hist_baseline[frame] + 0.05, y_hist_baseline[frame] + 0.05))
            #         text.set_text(f"Speed: {speed:.2f} m/s")
            #         artists.extend([line, marker, quiver, text])
            
            
            return artists

        from matplotlib import animation

        blit_flag = gif_path is None
        contour = None
        contourf = None
        anim = animation.FuncAnimation(
            fig_gif,
            update,
            init_func=init,
            frames=states.shape[0],
            interval=1000 / fps,
            blit=blit_flag,
        )
        writer = animation.PillowWriter(fps=fps, metadata=metadata)
        anim.save(gif_path, writer=writer, dpi=100)
        plt.close(fig_gif)


def generate_quarter_circle_reference(
    radius: float,
    num_points: int,
    target_speed: float,
    dt: float,
) -> np.ndarray:
    """Create ego reference states [x, vx, y, vy, z, vz] along a quarter circle ending at the gate."""

    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    # Uniform arc-length sampling so that consecutive reference points correspond to roughly target_speed * dt travel.
    delta_theta = (target_speed * dt) / radius
    if delta_theta <= 0.0:
        delta_theta = 0.0

    theta = np.arange(num_points, dtype=np.float64) * delta_theta
    theta = np.clip(theta, 0.0, np.pi / 2.0)

    # Quarter circle mirrored about the y-axis and reflected across y=x, starting near (-radius, -radius) and ending at the gate (0, 0).
    y = -radius * np.cos(theta)
    x = radius * np.sin(theta) - radius
    xy = np.stack([x, y], axis=1)

    tangents = np.gradient(xy, axis=0)
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norm[tangent_norm < 1e-8] = 1.0
    tangents /= tangent_norm

    velocities = target_speed * tangents
    reached_goal = theta >= (np.pi / 2.0)
    if np.any(reached_goal):
        first_goal_idx = int(np.argmax(reached_goal))
        tangent_dir = np.array([0.0, 1.0], dtype=np.float64)  # enforce exit direction along +y
        velocities[first_goal_idx] = target_speed * tangent_dir
        for i in range(first_goal_idx + 1, num_points):
            incremental_distance = target_speed * dt
            xy[i] = xy[i - 1] + tangent_dir * incremental_distance
            velocities[i] = target_speed * tangent_dir

    reference = np.zeros((num_points, 6), dtype=np.float64)
    reference[:, 0] = xy[:, 0]
    reference[:, 2] = xy[:, 1]
    reference[:, 4] = 0.0  # maintain constant altitude
    reference[:, 1] = velocities[:, 0]
    reference[:, 3] = velocities[:, 1]
    reference[:, 5] = 0.0  # vertical velocity remains zero
    return reference


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
    # parser.add_argument(
    #     "--speed",
    #     type=float,
    #     default=DroneRaceConfig.target_speed,
    #     help="Reference tangential speed in m/s (default: 1.5).",
    # )
    # parser.add_argument(
    #     "--radius",
    #     type=float,
    #     default=DroneRaceConfig.radius,
    #     help="Quarter-circle radius in metres (default: 2).",
    # )
    parser.add_argument(
        "--speed",
        type=float,
        default=MPCRaceConfig.target_speed,
        help="Reference tangential speed in m/s (default: 0.3).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=MPCRaceConfig.radius,
        help="Quarter-circle radius in metres (default: 2).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_CTRL_CFG.horizon,
        help=f"Prediction horizon for the MPC solver (default: {DEFAULT_CTRL_CFG.horizon}).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_CTRL_CFG.dt,
        help=f"Simulation timestep in seconds (default: {DEFAULT_CTRL_CFG.dt}).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_CTRL_CFG.gamma_cbf,
        help=f"CBF relaxation factor gamma (default: {DEFAULT_CTRL_CFG.gamma_cbf}).",
    )
    parser.add_argument(
        "--safety-radius",
        type=float,
        default=DEFAULT_CTRL_CFG.safety_radius,
        help=f"Minimum distance enforced between drones in metres (default: {DEFAULT_CTRL_CFG.safety_radius}).",
    )
    parser.add_argument(
        "--filter-weight",
        type=float,
        nargs=3,
        default=tuple(float(x) for x in np.asarray(DEFAULT_CTRL_CFG.filter_weight, dtype=np.float64)),
        help="Diagonal weights for the CBF safety filter (default: 1 1 1).",
    )
    parser.add_argument(
        "--filter-slack-weight",
        type=float,
        default=DEFAULT_CTRL_CFG.filter_slack_weight,
        help="Quadratic penalty weight on the CBF slack variable (default: 100).",
    )
    parser.add_argument(
        "--save-data",
        type=pathlib.Path,
        default=None,
        help="Optional path to save the rollout logs (npz). Defaults to MPCRaceConfig.save_path if omitted.",
    )
    parser.add_argument(
        "--static-opponents",
        type=int,
        nargs="*",
        default=(),
        help="Indices of opponent agents to keep stationary during the rollout.",
    )
    parser.add_argument(
        "--opponent-y-offsets",
        type=float,
        nargs="*",
        default=MPCRaceConfig.opponent_y_offsets,
        help="Initial y-offsets for opponent agents (default: -0.2 0.2).",
    )
    parser.add_argument(
        "--debug-cbf",
        action="store_true",
        help="Print per-step diagnostics for the CBF safety filter.",
    )
    return parser.parse_args()




# def main() -> None:
#     args = get_args()
#     args2 = parse_args()
#     env, policy_function = get_env_and_policy(args)

#     # import pdb; pdb.set_trace()

#     current_state = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
#     verif_reach_set_computer = ComputingVerifiedReachableSet(
#         current_state=current_state,
#     )
#     verified_set_deterministic, verified_set_scenario = verif_reach_set_computer.compute_verified_set(
#         args,
        
#         policy_function
#     )

#     # Plotting the verified reachable sets
#     x = np.arange(-0.9, 0.9, verif_reach_set_computer.epsilon_x)
#     y = np.arange(-2.6, 0, verif_reach_set_computer.epsilon_x)
#     X, Y = np.meshgrid(x, y)
#     V_det = verified_set_deterministic.value_function
#     # V_scen = verified_set_scenario.value_function

#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
#     fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

#     ax.plot_surface(X, Y, V_det, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)
#     ax.set_title("Verified Reachable Set (Deterministic)")
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
#     ax.set_zlabel("Value Function")

#     # ax2.plot_surface(X, Y, V_scen, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)
#     # ax2.set_title("Verified Reachable Set (Scenario)")
#     # ax2.set_xlabel("X Position")
#     # ax2.set_ylabel("Y Position")
#     # ax2.set_zlabel("Value Function")

#     # save figures
#     fig.savefig("verified_reachable_set_deterministic.png")
#     # fig2.savefig("verified_reachable_set_scenario.png")

def main2() -> None:
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)
    initial_state = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    # initial_state = np.array([0.0, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    # initial_state = np.array([-0.5, 0.0, -2.0, 0.7, 0.0, 0.0, 0.0, 0.0, -2.0, 0.3, 0.0, 0.0])
    # initial_state = np.array([0.1, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
#     initial_state = np.array([-0.53210269,  0.45827933, -2.31806132,  0.79822342, -0.00974961,  0.03249736,
#   0.79313229, -0.00653306, -2.24428504,  0.516076,    0.02628583, -0.05934889])
    # num_initial_conditions = 1
    # ego_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    # # ego_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    # ego_vx_init = np.zeros(num_initial_conditions)  
    # ego_y_init = np.random.uniform(-2.5, -1.5, size=num_initial_conditions)
    # ego_vy_init = np.random.uniform(0.7, 1.0, size=num_initial_conditions)
    # ego_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    # ego_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    # ad_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    # # ad_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    # ad_vx_init = np.zeros(num_initial_conditions)
    # ad_y_init = np.random.uniform(-2.5, -1.5, size=num_initial_conditions)
    # ad_vy_init = np.random.uniform(0.4, 0.7, size=num_initial_conditions)  # want ego to be faster than adversary for sake of target set relevance
    # ad_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    # ad_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    # Create a list of initial states for the simulations
    # initial_state = np.squeeze(np.array([ego_x_init, ego_vx_init, ego_y_init, ego_vy_init, ego_z_init, ego_vz_init,
    #                             ad_x_init, ad_vx_init, ad_y_init, ad_vy_init, ad_z_init, ad_vz_init]))
    print(f"Initial state for simulations: {initial_state}")
    print("--"*25)
    race_config = DroneRaceConfig(
        duration= 8, #5.5,
        initial_state=initial_state,
        value_path=args2.value_path,
    )
    mppi_config = DroneMPPIConfig(controlled_agent_index=args2.controlled_agent)
    mppi_config.opponent_gain = 0.5 #1.3 #0.5 #1.1  # initial estimate
    mppi_config.control_gain = 0.5

    # from dataclasses import fields
    # print(f"Config pre-switching controller: { {field.name: getattr(mppi_config, field.name) for field in fields(mppi_config)} }")
    # Compute offline verified reachable set
    verif_reach_set_computer = ComputingVerifiedReachableSet(
        current_state=initial_state,
    )
    # value_fn = (
    #         ReachabilityValueFunction.from_policy_path(
    #             str(reachability_value_path),
    #             device="cpu",
    #         )
    # )
    
    offline_verified_set, _ = verif_reach_set_computer.compute_verified_set(
        args,
        mppi_config,
        policy_function
    )
    print("Computed offline verified reachable set.")


    sim = DroneRaceSimulation(
        args=args,
        initial_state=initial_state,
        offline_verified_set=offline_verified_set,
        sim_cfg=race_config,
        mppi_cfg=mppi_config,
        reachability_value_path=args2.value_path,
        # reachability_policy=policy_function,
    )
    data = sim.run()

    # print(f"Config post-switching controller: { {field.name: getattr(mppi_config, field.name) for field in fields(mppi_config)} }")

    mppi_baseline_sim = DroneRaceMPPIBaselineSimulation(
        args=args,
        initial_state=initial_state,
        sim_cfg=race_config,
        mppi_cfg=mppi_config,
    )
    # print(f"Config for MPPI baseline: { {field.name: getattr(mppi_config, field.name) for field in fields(mppi_config)} }")
    baseline_data = mppi_baseline_sim.run()
    # import pdb; pdb.set_trace()
    plot_trajectories(
        states=data["state_log"],
        # modes=data["mode_log"],
        # expanded_regions=data["expanded_region_log"],
        sim=sim,
        expanded_regions=data["expanded_region_log"],
        modes=data["mode_log"],
        mppi_baseline_states=baseline_data["state_log"],
        save_path=args2.save_figure,
        gif_path=args2.save_gif,
        fps=20,
    )
    
def main3() -> None:
    # This main function will test the mppi cbf with a fixed opponent gain and no switching, to verify that the cbf is correctly implemented and can maintain safety under a known opponent behavior.
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)
    initial_state = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    # initial_state = np.array([0.1, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])


    race_config = DroneRaceConfig(
        duration= 8, #5.5,
        initial_state=initial_state,
        value_path=args2.value_path,
    )
    mppi_config = DroneMPPIConfig(controlled_agent_index=args2.controlled_agent)
    mppi_config.opponent_gain = 0.5 #1.3 #0.5 #1.1  # initial estimate
    mppi_config.control_gain = 0.5

    # # Compute offline verified reachable set
    # verif_reach_set_computer = ComputingVerifiedReachableSet(
    #     current_state=initial_state,
    # )
    # # value_fn = (
    # #         ReachabilityValueFunction.from_policy_path(
    # #             str(reachability_value_path),
    # #             device="cpu",
    # #         )
    # # )
    
    # offline_verified_set, _ = verif_reach_set_computer.compute_verified_set(
    #     args,
    #     mppi_config,
    #     policy_function
    # )
    # print("Computed offline verified reachable set.")


    # sim = DroneRaceSimulation(
    #     args=args,
    #     initial_state=initial_state,
    #     offline_verified_set=offline_verified_set,
    #     sim_cfg=race_config,
    #     mppi_cfg=mppi_config,
    #     reachability_value_path=args2.value_path,
    #     # reachability_policy=policy_function,
    # )
    # data = sim.run()

    # mppi_baseline_sim = DroneRaceMPPIBaselineSimulation(
    #     args=args,
    #     initial_state=initial_state,
    #     sim_cfg=race_config,
    #     mppi_cfg=mppi_config,
    # )
    # baseline_data = mppi_baseline_sim.run()
    # # import pdb; pdb.set_trace()

    mppi_cbf_cfg = MPPI_MPC_CBF_ControllerConfig(
        mppi_cfg = mppi_config,
    )
    # mppi_cbf_controller = MPPI_MPC_CBF_Controller(mppi_cbf_cfg)

    mppi_cbf_sim = DroneRaceMPPIBaselinewithCBFSimulation(
        args=args,
        initial_state=initial_state,
        sim_cfg=race_config,
        mppi_cbf_cfg=mppi_cbf_cfg,
    )
    data = mppi_cbf_sim.run()





    plot_trajectories2(
        states=data["state_log"],
        # modes=data["mode_log"],
        # expanded_regions=data["expanded_region_log"],
        sim=mppi_cbf_sim,
        modes=None,
        mppi_baseline_states=None, #baseline_data["state_log"],
        save_path=args2.save_figure,
        gif_path=args2.save_gif,
        fps=20,
    )
    
def main4() -> None:
    # This main function will test the mppi cbf with a fixed opponent gain and no switching, to verify that the cbf is correctly implemented and can maintain safety under a known opponent behavior.
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)
    # initial_state = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    initial_state = np.array([0.4, 0.0, -2.25, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])

    # args = parse_args()

    ctrl_cfg = DroneMPCConfig()
    # sim_cfg = DroneRaceConfig(
    #     # radius=1,
    # )
    # sim_cfg = MPCRaceConfig(
    # )
    # # static_indices = tuple(sorted(set(idx for idx in args.static_opponents if idx >= 0)))
    # sim = MPCSimulation(sim_cfg, ctrl_cfg)

    sim_cfg = MPCRaceConfig(
        radius=args2.radius,
        target_speed=args2.speed,
        duration=args2.duration,
        opponent_y_offsets=tuple(args2.opponent_y_offsets),
    )
    static_indices = tuple(sorted(set(idx for idx in args2.static_opponents if idx >= 0)))
    sim = MPCSimulation(sim_cfg, ctrl_cfg, static_opponents=static_indices, debug=args2.debug_cbf)


    data = sim.run()

    plot_trajectories2(
        states=data["states"],
        # modes=data["mode_log"],
        # expanded_regions=data["expanded_region_log"],
        sim=sim,
        modes=None,
        mppi_baseline_states=None, #baseline_data["state_log"],
        save_path=args2.save_figure,
        gif_path=args2.save_gif,
        fps=20,
    )

def main5() -> None:
    # This main function will conduct a monte carlo simulation to evaluate the performance of
    # Hybrid Controller, MPPI baseline, and MPPI with CBF across a range of initial conditions
    # with fixed opponent behavior. Trajectories will be saved to a file for later analysis and plotting.
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)

    # Define a set of initial conditions for the ego agent (sample in all 6 state dimensions)
    num_initial_conditions = 500
    ego_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    ego_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    ego_y_init = np.random.uniform(-2.5, -0.5, size=num_initial_conditions)
    ego_vy_init = np.random.uniform(0.7, 1.0, size=num_initial_conditions)
    ego_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    ego_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    ad_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    ad_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    ad_y_init = np.random.uniform(-2.5, -0.5, size=num_initial_conditions)
    ad_vy_init = np.random.uniform(0.4, 0.7, size=num_initial_conditions)  # want ego to be faster than adversary for sake of target set relevance
    ad_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    ad_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    # Create a list of initial states for the simulations
    initial_states = np.stack([ego_x_init, ego_vx_init, ego_y_init, ego_vy_init, ego_z_init, ego_vz_init,
                                ad_x_init, ad_vx_init, ad_y_init, ad_vy_init, ad_z_init, ad_vz_init], axis=1)
    
    # Create a file path for saving the monte carlo results
    monte_carlo_save_path = "monte_carlo_results.npz" 


    # Loop through initial states and run simulations for each controller type, saving trajectories to file
    # from tqdm import tqdm
    all_data_hybrid = []
    all_data_mppi_baseline = []
    all_data_mppi_cbf = []
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
        mppi_config.opponent_gain = 0.5
        mppi_config.control_gain = 0.5
        verif_reach_set_computer = ComputingVerifiedReachableSet(
            current_state=initial_state,
        )
        offline_verified_set, _ = verif_reach_set_computer.compute_verified_set(
            args,
            mppi_config,
            policy_function
        )
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

        # Run MPPI baseline
        sim_mppi_baseline = DroneRaceMPPIBaselineSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cfg=mppi_config,
        )
        data_mppi_baseline = sim_mppi_baseline.run()
        all_data_mppi_baseline.append(data_mppi_baseline)

        # Run MPPI with CBF
        mppi_cbf_cfg = MPPI_MPC_CBF_ControllerConfig(
            mppi_cfg = mppi_config,
        )
        sim_mppi_cbf = DroneRaceMPPIBaselinewithCBFSimulation(
            args=args,
            initial_state=initial_state,
            sim_cfg=race_config,
            mppi_cbf_cfg=mppi_cbf_cfg,
        )
        data_mppi_cbf = sim_mppi_cbf.run()
        all_data_mppi_cbf.append(data_mppi_cbf)

        # Save intermediate results to file every 25 simulations
        if len(all_data_hybrid) % 10 == 0:
            np.savez(monte_carlo_save_path, hybrid=all_data_hybrid, mppi_baseline=all_data_mppi_baseline, mppi_cbf=all_data_mppi_cbf)

def main6() -> None:
    # This main function will conduct a monte carlo simulation to evaluate the performance of
    # Hybrid Controller, MPPI baseline, and MPPI with CBF across a range of initial conditions
    # with fixed opponent behavior. Trajectories will be saved to a file for later analysis and plotting.
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)

    # Define a set of initial conditions for the ego agent (sample in all 6 state dimensions)
    num_initial_conditions = 500
    ego_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    # ego_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    ego_vx_init = np.zeros(num_initial_conditions)  
    ego_y_init = np.random.uniform(-2.5, -1.5, size=num_initial_conditions)
    ego_vy_init = np.random.uniform(0.7, 1.0, size=num_initial_conditions)
    ego_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    ego_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    ad_x_init = np.random.uniform(-0.8, 0.8, size=num_initial_conditions)
    # ad_vx_init = np.random.uniform(-0.5, 0.5, size=num_initial_conditions)
    ad_vx_init = np.zeros(num_initial_conditions)
    ad_y_init = np.random.uniform(-2.5, -1.5, size=num_initial_conditions)
    ad_vy_init = np.random.uniform(0.4, 0.7, size=num_initial_conditions)  # want ego to be faster than adversary for sake of target set relevance
    ad_z_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)
    ad_vz_init = np.random.uniform(-0.1, 0.1, size=num_initial_conditions)

    # Create a list of initial states for the simulations
    initial_states = np.stack([ego_x_init, ego_vx_init, ego_y_init, ego_vy_init, ego_z_init, ego_vz_init,
                                ad_x_init, ad_vx_init, ad_y_init, ad_vy_init, ad_z_init, ad_vz_init], axis=1)
    
    # Create a file path for saving the monte carlo results
    # monte_carlo_save_path = "monte_carlo_results_additional.npz" 
    # monte_carlo_save_path = "monte_carlo_results_no_learned_policy_ablation.npz"
    monte_carlo_save_path = f"monte_carlo_results_all_controllers_seed_{seed}.npz"


    # Loop through initial states and run simulations for each controller type, saving trajectories to file
    # from tqdm import tqdm
    all_data_hybrid = []
    all_data_mppi_baseline = []
    all_data_mppi_cbf = []
    all_data_mppi_safe_baseline = []
    all_data_mppi_warmstart_learned_baseline = []
    all_data_policy_only_baseline = []
    all_data_hybrid_no_learned_policy = []
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
        mppi_config.opponent_gain = 0.5
        mppi_config.control_gain = 0.5
        verif_reach_set_computer = ComputingVerifiedReachableSet(
            current_state=initial_state,
        )
        offline_verified_set, _ = verif_reach_set_computer.compute_verified_set(
            args,
            mppi_config,
            policy_function
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

        ####

        # Run switching controller without learned policy in switching logic (ablation)
        ## debugging
        # initial_state_debug = np.array([-0.31192528, -0.18630495, -1.49892508,  0.80690527,  0.07623354,  0.03269144, 
        #                                 -0.60761344, -0.02786871, -0.75642083,  0.57885821,  0.02777298,  0.00635087])
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
        )
        sim_mppi_baseline_safe.mppi_cfg.safety_weight = 5.0
        data_mppi_baseline_safe = sim_mppi_baseline_safe.run()
        all_data_mppi_safe_baseline.append(data_mppi_baseline_safe)

        # Run MPPI with CBF
        mppi_cbf_cfg = MPPI_MPC_CBF_ControllerConfig(
            mppi_cfg = mppi_config,
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

        # # Save intermediate results to file every 25 simulations
        # if len(all_data_mppi_safe_baseline) % 10 == 0:
        #     np.savez(monte_carlo_save_path,
        #                 hybrid=all_data_hybrid, 
        #                 # hybrid_no_learned_policy=all_data_hybrid_no_learned_policy,
        #                 mppi_baseline=all_data_mppi_baseline, 
        #                 mppi_safe_baseline=all_data_mppi_safe_baseline, 
        #                 mppi_cbf=all_data_mppi_cbf, 
        #                 mppi_warmstart_learned_baseline=all_data_mppi_warmstart_learned_baseline,
        #                 policy_only_baseline=all_data_policy_only_baseline)

        # Save intermediate results to file every 10 simulations
        if len(all_data_hybrid_no_learned_policy) % 10 == 0:
            np.savez(monte_carlo_save_path,
                        hybrid=all_data_hybrid, 
                        hybrid_no_learned_policy=all_data_hybrid_no_learned_policy,
                        mppi_baseline=all_data_mppi_baseline, 
                        mppi_safe_baseline=all_data_mppi_safe_baseline, 
                        mppi_cbf=all_data_mppi_cbf, 
                        mppi_warmstart_learned_baseline=all_data_mppi_warmstart_learned_baseline,
                        policy_only_baseline=all_data_policy_only_baseline
                        )





if __name__ == "__main__":
    # main()
    main2()
    # main3()
    # main4()
    # main5()
    # main6()






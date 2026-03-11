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

# from env_utils import NoResetSyncVectorEnv, evaluate_V_batch, find_a_batch, find_a, get_args, get_env_and_policy
from env_utils_ppo import get_args_ppo, get_env_and_policy_ppo, find_a_ppo

# from intent_estimation_utils import ControlGainEstimator
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm
import cvxpy as cp

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
            raise ValueError(f"{name} must be non-negative")

class SafetyFilter(DroneMPPIController):
    """CBF Safety Filter that projects a proposed control input to the closest control input that satisfies the CBF constraint."""
    def __init__(self, config: MPPI_MPC_CBF_ControllerConfig) -> None:
        super().__init__(config.mppi_cfg)
        self.cbf_config = config
        self.gamma_cbf = config.gamma_cbf
        self.max_control = config.max_control
        self.filter_weight = np.diag(config.filter_weight)  # weight for the CBF safety filter in the optimization objective
        self.filter_slack_weight = config.filter_slack_weight  # weight for the slack variable in the optimization objective
        solver_name = config.solver #.lower()
        if not hasattr(cp, solver_name):
            raise ValueError(f"Solver {config.solver} is not available in cvxpy.")
        self._solver = getattr(cp, solver_name)
        self._solver_options = dict(config.solver_options)

        self.A = np.eye(config.mppi_cfg.per_agent_state_dim)
        dt = config.mppi_cfg.dt
        control_gain = config.mppi_cfg.control_gain
        self.A[0, 1] = dt
        self.A[2, 3] = dt
        self.A[4, 5] = dt

        self.B = np.zeros((config.mppi_cfg.per_agent_state_dim, config.mppi_cfg.per_agent_control_dim))
        self.B[1, 0] = control_gain * dt
        self.B[3, 1] = control_gain * dt
        self.B[5, 2] = control_gain * dt
        half_dt_sq_gain = 0.5 * dt * dt * control_gain
        self.B[0, 0] = half_dt_sq_gain
        self.B[2, 1] = half_dt_sq_gain
        self.B[4, 2] = half_dt_sq_gain

        self.position_indices = np.array([0, 2, 4], dtype=np.int64)
        self.velocity_indices = np.array([1, 3, 5], dtype=np.int64)

        self.Qp = np.diag(config.position_weight)
        self.Rv = np.diag(config.velocity_weight)
        self.Ru = np.diag(config.control_weight)
        self.Qp_T = np.diag(config.terminal_position_weight)
        self.Rv_T = np.diag(config.terminal_velocity_weight)

    def _predict_opponent_positions_pid_control(self, state: np.ndarray) -> Dict[int, np.ndarray]:
        predictions: Dict[int, np.ndarray] = {}
        predictions_velocity: Dict[int, np.ndarray] = {}
        if not self.other_agent_indices:
            return predictions

        dt = self.config.dt
        horizon = self.config.horizon
        for agent_idx in self.other_agent_indices:
            feedback = super(). _opponent_feedback(state, agent_idx)
            accel = self.config.opponent_gain * feedback
            block = state[self._agent_slice(agent_idx)]
            pos = np.zeros((horizon + 1, 3), dtype=np.float64)
            vel = np.zeros((horizon + 1, 3), dtype=np.float64)
            pos[0] = block[self.position_indices]
            vel[0] = block[self.velocity_indices]
            for t in range(horizon):
                pos[t + 1] = pos[t] + dt * vel[t]
                vel[t + 1] = vel[t] + dt * accel
            predictions[agent_idx] = pos
            predictions_velocity[agent_idx] = vel
        
        return predictions, predictions_velocity
    
    def compute_safe_control(
        self,
        controlled_block: np.ndarray,
        u_nominal: np.ndarray,
        opponent_predictions: Dict[int, np.ndarray],
        opponent_velocities: Dict[int, np.ndarray],
        nominal_states: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Compute a safe control action using a CBF-based safety filter.

        Args:
            controlled_block: The current state of the controlled agent (shape: [state_dim])
            u_nominal: The nominal control action from MPPI (shape: [control_dim])
            opponent_predictions: A dictionary mapping agent indices to their predicted trajectories (shape: [num_steps, state_dim])
            opponent_velocities: A dictionary mapping agent indices to their predicted velocities (shape: [num_steps, velocity_dim])
            nominal_states: The nominal state trajectory from MPPI (shape: [num_steps, state_dim])
        Returns:
            A tuple containing:
                - The safe control action (shape: [control_dim])
                - A dictionary of additional information (e.g., whether the CBF constraint was active)
        """
        if not self.other_agent_indices:
            # No other agents, so no safety constraints needed
            # print(f"CBF filter skipped: no other agents detected.")
            return u_nominal.copy(), {"status": "NO_CONSTRAINTS", "filtered": False}

        if nominal_states.shape[0] < 2:
            # Not enough states to compute the CBF constraint
            # print(f"CBF filter skipped: insufficient nominal trajectory length ({nominal_states.shape[0]} steps).")
            return u_nominal.copy(), {"status": "INSUFFICIENT_TRAJ", "filtered": False}

        # radius_sq = self.cbf_config.mppi_cfg.safety_radius ** 2
        # radius = (1 + max(0, (opp))
        # print(f"Computing CBF filter: radius_sq={radius_sq}, gamma={self.gamma_cbf}, max_control={self.max_control}")
        gamma = self.gamma_cbf

        u_var = cp.Variable(self.control_dim)
        slack = cp.Variable(nonneg=True)

        # print(f"shape of u_var: {u_var.shape}, shape of u_nominal: {u_nominal.shape}, filter_weight: {self.filter_weight}, filter_slack_weight: {self.filter_slack_weight}")
        objective = 0.5 * cp.quad_form(u_var - u_nominal, self.filter_weight) + 0.5 * self.filter_slack_weight * cp.square(slack)
        constraints = [cp.abs(u_var) <= self.max_control, slack >= 0]

        next_state_expr = self.A @ controlled_block + self.B @ u_var
        p_next_expr = next_state_expr[self.position_indices]
        anchor_pos_next = nominal_states[1, self.position_indices]
        anchor_pos_cur = nominal_states[0, self.position_indices]
        # print(f"Anchor position (current): {anchor_pos_cur}, Anchor position (next): {anchor_pos_next}")
        active_constraints = 0
        min_violation = float("inf")

        nominal_next = self.A @ controlled_block + self.B @ u_nominal
        constraint_details = []

        for agent_index in self.other_agent_indices:
            predicted_pos = opponent_predictions.get(agent_index)
            # print(f"Agent {agent_index}: predicted_pos: {predicted_pos if predicted_pos is not None else None}")
            if predicted_pos is None or predicted_pos.shape[0] < 2:
                continue  # Skip if no valid prediction for this agent
            
            obs_pos_cur = predicted_pos[0]
            obs_pos_next = predicted_pos[1]

            obs_vel_cur = opponent_velocities[agent_index][0] if agent_index in opponent_velocities else np.zeros(3, dtype=np.float64)
            # obs_vel_next = opponent_velocities[agent_index][1] if agent_index in opponent_velocities else np.zeros(3, dtype=np.float64)

            diff_next = anchor_pos_next - obs_pos_next
            grad = 2.0 * diff_next
            if np.linalg.norm(grad) < 1e-6:
                # print(f"Warning: near-zero gradient for agent {agent_index} in CBF constraint. Adding small regularization to avoid numerical issues.")
                grad = 2.0 * (diff_next + 1e-3 * np.array([1.0, 0.0, 0.0]))
            radius = (1 + max(0, (obs_vel_cur[-1] - controlled_block[self.velocity_indices][-1]))) * self.cbf_config.mppi_cfg.safety_radius
            radius_sq = radius ** 2
            # print(f"CBF constraint for agent {agent_index}: radius={radius:.4f}, radius_sq={radius_sq:.4f}")
            h_bar = float(np.dot(diff_next, diff_next) - radius_sq)

            diff_cur = anchor_pos_cur - obs_pos_cur
            h_current = float(np.dot(diff_cur, diff_cur) - radius_sq)
            min_violation = min(min_violation, h_current)

            linearized = grad @ (p_next_expr - anchor_pos_next) + h_bar
            lower_bound = (1.0 - gamma) * h_current
            constraints.append(linearized + slack >= lower_bound)
            active_constraints += 1

            nominal_linearized = float(grad @ (nominal_next[self.position_indices] - anchor_pos_next) + h_bar)

            detail = {
                "agent_index": int(agent_index),
                "h_current": h_current,
                "h_bar": h_bar,
                "lower_bound": lower_bound,
                "grad_norm": float(np.linalg.norm(grad)),
                "linearized_nominal": nominal_linearized,
            }
            constraint_details.append(detail)
            

        if active_constraints == 0:
            # print(f"CBF filter optimization skipped: no active constraints. min_violation={min_violation:.4f}")
            return u_nominal.copy(), {"status": "NO_ACTIVE_CONSTRAINTS", "filtered": False, "constraints": [], "slack": 0.0}

        problem = cp.Problem(cp.Minimize(objective), constraints)
        solve_kwargs = dict(warm_start=True, **self._solver_options)

        try:
            # print(f"Solving CBF filter optimization with solver {self._solver} with {active_constraints} active constraints and min violation {min_violation:.4f}.")
            problem.solve(solver=self._solver, **solve_kwargs)
        except cp.error.SolverError:
            # print(f"Solver {self._solver} failed with options {self._solver_options}. Falling back to ECOS solver.")
            problem.solve(solver=cp.ECOS, warm_start=True)

        status = problem.status
        optimal_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        feasible = status in optimal_statuses

        if not feasible or u_var.value is None:
            # print(f"CBF filter optimization failed (status: {status}). Applying nominal control.")
            return np.clip(u_nominal, -self.max_control, self.max_control), {
                "status": status,
                "filtered": False,
                "violation": min_violation,
                "constraints": constraint_details,
                "slack": float(slack.value) if slack.value is not None else float("nan"),
            }
        
        filtered_control = np.asarray(u_var.value, dtype=np.float64)
        filtered_control = np.clip(filtered_control, -self.max_control, self.max_control)
        filtered = not np.allclose(filtered_control, u_nominal, atol=1e-4)

        
        return filtered_control, {
            "status": status,
            "filtered": filtered,
            "violation": min_violation,
            "constraints": constraint_details,
            "slack": float(slack.value) if slack.value is not None else float("nan"),
        }
    
    def solve_total(
        self,
        state: np.ndarray,
        ppo_policy,
    ) -> Tuple[np.ndarray, Dict[str, object], np.ndarray]:
        """
        Compute the control action by first getting the nominal control from learned policy and then applying the CBF safety filter.
        Args:
            state: The current state of the controlled agent (shape: [state_dim])
            ppo_policy: The learned policy to compute the nominal control action
        Returns:
            A tuple containing:
                - The safe control action after applying the CBF filter (shape: [control_dim])
                - A dictionary of additional information from the safety filter
                - The nominal control action from the learned policy (shape: [control_dim])
        """
        nominal_control = find_a_ppo(state, ppo_policy)
        opponent_predictions, opponent_velocities = self._predict_opponent_positions_pid_control(state)
        next_state, _ = self.simulate_step(state, nominal_control)
        nominal_states = np.vstack((state, next_state))

        controlled_block = np.asarray(state[self._agent_slice(self.controlled_agent)], dtype=np.float64)
        safe_control, info = self.compute_safe_control(
            controlled_block,
            nominal_control,
            opponent_predictions,
            opponent_velocities,
            nominal_states
        )
        return safe_control, info, nominal_control
        

     

    
class DroneRacePPOCBFSimulation:
    """Simulates an ego drone trying to overtake another drone using a learned policy baseline without any MPPI or verified set."""

    def __init__(
            self,
            args,
            initial_state: np.ndarray,
            sim_cfg: DroneRaceConfig,
            mppi_cbf_cfg: MPPI_MPC_CBF_ControllerConfig,
            # policy_function: ReachabilityValueFunction
            # reachability_value_path: Optional[pathlib.Path] = None
            ppo_policy,
    ) -> None:
        self.sim_cfg = sim_cfg
        # self.policy_function = policy_function.policy
        # value_fn = (
        #     ReachabilityValueFunction.from_policy_path(
        #         str(reachability_value_path),
        #         device="cpu",
        #     )
        #     if reachability_value_path is not None
        #     else None
        # )
        # self.policy_function = value_fn.policy
        self.mppi_cfg = deepcopy(mppi_cbf_cfg)
        self.mppi_cfg.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the learned policy baseline
        self.ppo_policy = ppo_policy
        # self.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the learned policy baseline
        # self.mppi_cfg.control_gain = 0.5  # assume a fixed control gain for the learned policy baseline
        # self.mppi_controller = DroneMPPIController(self.mppi_cfg)  # only used for simulate_step, not for control in this baseline
        self.controller = SafetyFilter(self.mppi_cfg)  # use the CBF safety filter as the controller to apply the learned policy with safety filtering
        self.dt = 0.1  # assume same time step as MPPI for fair comparison
        self.num_steps = int(np.ceil(sim_cfg.duration / self.dt))
        self.state = initial_state
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, 3), dtype=np.float64)  # assume control dimension of 3

    # def find_a(self, state):
    #         tmp_obs = np.array(state).reshape(1,-1)
    #         tmp_batch = Batch(obs = tmp_obs, info = Batch())
    #         tmp = self.policy_function(tmp_batch, model = "actor_old").act
    #         act = self.policy_function.map_action(tmp).cpu().detach().numpy().flatten()
    #         return act
    

    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""
        for t in range(self.num_steps):
            # reset_nominal = False
            # action = self.find_a(self.state)
            action, _, _ = self.controller.solve_total(self.state, self.ppo_policy)
            reached_goal = self.state[2] > 0.0 and np.abs(self.state[0]) <= 0.3
            if reached_goal:
                self.state_log[t:] = self.state  # log current state at the time of reaching goal
                self.control_log[t:] = action[:3]  # log control input at the time of reaching goal
                break
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            next_state, _ = self.controller.simulate_step(self.state, action)
            self.state = next_state

        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
        }
"""MPPI controller for drone racing with CBF safety filtering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from mppi_mpc_controller import DroneMPPIController, DroneMPPIConfig


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

class MPPI_MPC_CBF_Controller(DroneMPPIController):
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

        self._ref_positions: Optional[np.ndarray] = None
        self._ref_velocities: Optional[np.ndarray] = None
        self._last_state_traj: Optional[np.ndarray] = None
        self._last_control_seq: Optional[np.ndarray] = None

    def _predict_opponent_positions_no_control(self, state: np.ndarray) -> Dict[int, np.ndarray]:
        predictions: Dict[int, np.ndarray] = {}
        if not self.other_agent_indices:
            return predictions

        dt = self.config.dt
        horizon = self.config.horizon
        for agent_idx in self.other_agent_indices:
            block = state[self._agent_slice(agent_idx)]
            pos = np.zeros((horizon + 1, 3), dtype=np.float64)
            vel = np.zeros((horizon + 1, 3), dtype=np.float64)
            pos[0] = block[self.position_indices]
            vel[0] = block[self.velocity_indices]
            for t in range(horizon):
                pos[t + 1] = pos[t] + dt * vel[t]
                vel[t + 1] = vel[t]
            predictions[agent_idx] = pos
        return predictions

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
            # print(f"distance: {np.linalg.norm(anchor_pos_cur - obs_pos_cur)}")
            # print(f"radius: {np.sqrt(radius_sq)}")
            # print(f"h_current: {h_current:.4f}, h_bar: {h_bar:.4f}, lower_bound: {lower_bound:.4f}")
            # print(f"nominal linearized: {nominal_linearized:.4f}")
        
            

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

        # import pdb; pdb.set_trace()
        # print(f"CBF filter optimization status: {status}, filtered: {filtered}, min_violation: {min_violation:.4f}, slack: {slack.value:.4f}")


        # printing for debugging
        # print(f"distance: {np.linalg.norm(anchor_pos_cur - obs_pos_cur)}")
        # print(f"radius: {np.sqrt(radius_sq)}")
        # print(f"h_current: {h_current:.4f}, h_bar: {h_bar:.4f}, lower_bound: {lower_bound:.4f}")
        # print(f"nominal linearized: {nominal_linearized:.4f}")
        # print(f"slack weight: {self.filter_slack_weight}, slack value: {slack.value:.4f}")
        # print(f"filtered control: {filtered_control}, nominal control: {u_nominal}, difference: {filtered_control - u_nominal}")
        # print(f"Filter activated: {filtered}, ")

        # import pdb; pdb.set_trace()
        return filtered_control, {
            "status": status,
            "filtered": filtered,
            "violation": min_violation,
            "constraints": constraint_details,
            "slack": float(slack.value) if slack.value is not None else float("nan"),
        }
    
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
        margin = self._safety_margin(state)
        if margin < 0.0:
            cost += self.config.safety_weight * margin * margin
        return cost, margin

    def solve_total(
        self,
        initial_state: Sequence[float],
        reference: np.ndarray,
        reset_nominal: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Solve for the optimal control sequence using MPPI and apply the CBF safety filter to the first control action.
        Args:
            initial_state: The current state of the system (shape: [state_dim])
            reference: The reference trajectory for MPPI (shape: [num_steps, state_dim])
            reset_nominal: Whether to reset the nominal trajectory in MPPI
        Returns:
            A tuple containing:
                - The safe control action to apply (shape: [control_dim])
                - A dictionary of additional information (e.g., whether the CBF constraint was active)
        """
        nominal_control, info = super().solve(initial_state, reference, reset_nominal)
        ######
        state = np.asarray(initial_state, dtype=np.float64)
        # if state.shape != (self.state_dim,):
        #     raise ValueError(f"initial_state must have shape ({self.state_dim},)")

        # if reference is not None:
        #     self.update_reference(reference)
        # elif self._ref_positions is None or self._ref_velocities is None:
        #     raise ValueError("Reference trajectory has not been set")

        # controlled_block = state[self._agent_slice(self.controlled_agent)].copy()

        # self._ensure_nominal(controlled_block)


        
        # nominal_result = self._solve_nominal_mpc(controlled_block)
        # if not nominal_result["feasible"]:
        #     info = {
        #         "mpc_status": nominal_result["status"],
        #         "mpc_feasible": False,
        #         "objective": float("nan"),
        #         "filter_status": "SKIPPED",
        #         "filtered": False,
        #         "nominal_control": nominal_result["fallback"],
        #         "applied_control": nominal_result["fallback"],
        #     }
        #     return nominal_result["fallback"].copy(), info

        # u_nom = nominal_result["control_traj"][0]
        # print(f"Nominal control from MPC: {u_nom}")

        # filter_control, filter_info = self.compute_safe_control(
        #     controlled_block=controlled_block,
        #     u_nominal=u_nom,
        #     # opponent_predictions=self._predict_opponent_positions_pid_control(state),
        #     opponent_predictions=self._predict_opponent_positions_no_control(state),
        #     nominal_states=nominal_result["state_traj"],
        # )
        # info: Dict[str, np.ndarray | float | str | bool] = {
        #     "mpc_status": nominal_result["status"],
        #     "mpc_feasible": True,
        #     "objective": nominal_result["objective"],
        #     "state_traj": nominal_result["state_traj"],
        #     "control_traj": nominal_result["control_traj"],
        #     "nominal_control": u_nom,
        #     "applied_control": filter_control,
        #     "filter_status": filter_info["status"],
        #     "filtered": filter_info["filtered"],
        # }
        # if "violation" in filter_info:
        #     info["violation"] = filter_info["violation"]
        # if "constraints" in filter_info:
        #     info["cbf_constraints"] = filter_info["constraints"]
        # if "slack" in filter_info:
        #     info["cbf_slack"] = filter_info["slack"]

        # return filter_control, info, nominal_result["control_traj"][0].copy()
        ######
        opponent_predictions, opponent_velocities = self._predict_opponent_positions_pid_control(state)
        # opponent_predictions = self._predict_opponent_positions_no_control(np.asarray(initial_state, dtype=np.float64))

        # need to simulate next state based on nominal control to compute the CBF constraint, since it depends on the next state
        next_state, _ = self.simulate_step(initial_state, nominal_control)
        # print(f"Initial state: {initial_state.shape}, Next state: {next_state.shape}, Nominal control: {nominal_control.shape}")
        nominal_states = np.vstack((initial_state, next_state))
        
        controlled_block = np.asarray(initial_state[self._agent_slice(self.controlled_agent)], dtype=np.float64)
        safe_control, cbf_info = self.compute_safe_control(
            controlled_block=controlled_block,
            u_nominal=nominal_control,
            opponent_predictions=opponent_predictions,
            opponent_velocities=opponent_velocities,
            # nominal_states=info["best_traj"],
            nominal_states=nominal_states,
        )
        combined_info = {**info, **cbf_info}
        # import pdb; pdb.set_trace()
        return safe_control, combined_info, nominal_control
    

    def update_reference(self, reference: np.ndarray) -> None:
        self._set_reference(reference)

    def _ensure_nominal(self, controlled_block: np.ndarray) -> None:
        if self._last_state_traj is not None and self._last_control_seq is not None:
            return

        horizon = self.config.horizon
        traj = np.zeros((horizon + 1, self.per_agent_state_dim), dtype=np.float64)
        traj[0] = controlled_block
        for t in range(horizon):
            traj[t + 1] = self.A @ traj[t]

        controls = np.zeros((horizon, self.control_dim), dtype=np.float64)

        self._last_state_traj = traj
        self._last_control_seq = controls

    def _update_warm_start(self, state_traj: np.ndarray, control_traj: np.ndarray) -> None:
        horizon = self.config.horizon
        shifted_states = state_traj.copy()
        shifted_controls = control_traj.copy()

        if horizon >= 1:
            shifted_states[:-1] = state_traj[1:]
            shifted_states[-1] = state_traj[-1]
            if horizon > 1:
                shifted_controls[:-1] = control_traj[1:]
            shifted_controls[-1] = control_traj[-1]

        self._last_state_traj = shifted_states
        self._last_control_seq = shifted_controls

    def _set_reference(self, reference: np.ndarray) -> None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim != 2:
            raise ValueError("reference must be a 2-D array")
        if ref.shape[1] not in (3, self.per_agent_state_dim):
            raise ValueError("reference must provide 3 (positions) or 6 (full state) columns")
        if ref.shape[0] == 0:
            raise ValueError("reference must contain at least one waypoint")

        horizon = self.config.horizon
        stage_count = min(ref.shape[0], horizon)

        positions = np.zeros((stage_count, 3), dtype=np.float64)
        velocities = np.zeros_like(positions)

        if ref.shape[1] == 3:
            positions[:stage_count] = ref[:stage_count, :3]
        else:
            positions[:stage_count] = ref[:stage_count, [0, 2, 4]]
            velocities[:stage_count] = ref[:stage_count, [1, 3, 5]]

        if stage_count < horizon:
            last_pos = positions[-1].copy()
            last_vel = velocities[-1].copy()
            pos_list = [positions]
            vel_list = [velocities]
            for _ in range(horizon - stage_count):
                last_pos = last_pos + self.config.dt * last_vel
                pos_list.append(last_pos[None, :])
                vel_list.append(last_vel[None, :])
            positions = np.vstack(pos_list)
            velocities = np.vstack(vel_list)

        self._ref_positions = np.vstack([positions, positions[-1]])
        self._ref_velocities = np.vstack([velocities, velocities[-1]])






# if __name__ == "__main__":


        

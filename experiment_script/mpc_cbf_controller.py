"""Receding-horizon MPC with a single-step CBF safety filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np


@dataclass
class DroneMPCConfig:
    """Configuration for nominal tracking MPC and the CBF safety filter."""

    horizon: int = 15
    dt: float = 0.1
    control_gain: float = 5.0
    position_weight: Sequence[float] = (5.0, 5.0, 5.0)
    velocity_weight: Sequence[float] = (1.0, 1.0, 1.0)
    control_weight: Sequence[float] = (0.1, 0.1, 0.1)
    terminal_position_weight: Sequence[float] = (10.0, 10.0, 10.0)
    terminal_velocity_weight: Sequence[float] = (2.0, 2.0, 2.0)
    filter_weight: Sequence[float] = (1.0, 1.0, 1.0)
    filter_slack_weight: float = 100000.0
    gamma_cbf: float = 0.6
    safety_radius: float = 0.3
    max_control: float | Sequence[float] = 2.5
    num_agents: int = 3
    per_agent_state_dim: int = 6
    per_agent_control_dim: int = 3
    controlled_agent_index: int = 0
    solver: str = "OSQP"
    solver_options: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 < self.gamma_cbf <= 1.0:
            raise ValueError("gamma_cbf must lie within (0, 1]")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.per_agent_state_dim != 6:
            raise ValueError("per_agent_state_dim must be 6 for the drone model")
        if self.per_agent_control_dim != 3:
            raise ValueError("per_agent_control_dim must be 3 for the drone model")
        if not 0 <= self.controlled_agent_index < self.num_agents:
            raise ValueError("controlled_agent_index out of range")

        self.position_weight = self._to_weight(self.position_weight, name="position_weight")
        self.velocity_weight = self._to_weight(self.velocity_weight, name="velocity_weight")
        self.control_weight = self._to_weight(self.control_weight, name="control_weight")
        self.terminal_position_weight = self._to_weight(
            self.terminal_position_weight, name="terminal_position_weight"
        )
        self.terminal_velocity_weight = self._to_weight(
            self.terminal_velocity_weight, name="terminal_velocity_weight"
        )
        self.max_control = self._to_weight(self.max_control, name="max_control", allow_scalar=True)
        self.filter_weight = self._to_weight(self.filter_weight, name="filter_weight", allow_scalar=True)
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
            arr = np.repeat(float(arr), self.per_agent_control_dim)
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must contain exactly three entries")
        if np.any(arr < 0):
            raise ValueError(f"{name} entries must be non-negative")
        return arr


class DroneMPCCBFController:
    """Nominal MPC plus a single-step CBF filter for collision avoidance."""

    def __init__(self, config: DroneMPCConfig) -> None:
        self.config = config
        self.position_indices = np.array([0, 2, 4], dtype=np.int64)
        self.velocity_indices = np.array([1, 3, 5], dtype=np.int64)

        self.num_agents = config.num_agents
        self.state_dim = self.num_agents * config.per_agent_state_dim
        self.per_agent_state_dim = config.per_agent_state_dim
        self.control_dim = config.per_agent_control_dim
        self.controlled_agent = config.controlled_agent_index
        self.other_agent_indices = [idx for idx in range(self.num_agents) if idx != self.controlled_agent]

        self.A = np.eye(config.per_agent_state_dim)
        dt = config.dt
        control_gain = config.control_gain
        self.A[0, 1] = dt
        self.A[2, 3] = dt
        self.A[4, 5] = dt

        self.B = np.zeros((config.per_agent_state_dim, config.per_agent_control_dim), dtype=np.float64)
        half_dt_sq_gain = 0.5 * dt * dt * control_gain
        self.B[0, 0] = half_dt_sq_gain
        self.B[2, 1] = half_dt_sq_gain
        self.B[4, 2] = half_dt_sq_gain
        self.B[1, 0] = dt * control_gain
        self.B[3, 1] = dt * control_gain
        self.B[5, 2] = dt * control_gain

        self.Qp = np.diag(config.position_weight)
        self.Rv = np.diag(config.velocity_weight)
        self.Ru = np.diag(config.control_weight)
        self.Qp_T = np.diag(config.terminal_position_weight)
        self.Rv_T = np.diag(config.terminal_velocity_weight)
        self.filter_weight = np.diag(config.filter_weight)
        self.filter_slack_weight = float(config.filter_slack_weight)

        solver_name = config.solver.upper()
        if not hasattr(cp, solver_name):
            raise ValueError(f"Unsupported solver '{config.solver}' for cvxpy")
        self._solver = getattr(cp, solver_name)
        self._solver_options = dict(config.solver_options)

        self._ref_positions: Optional[np.ndarray] = None
        self._ref_velocities: Optional[np.ndarray] = None
        self._last_state_traj: Optional[np.ndarray] = None
        self._last_control_seq: Optional[np.ndarray] = None

    def update_reference(self, reference: np.ndarray) -> None:
        self._set_reference(reference)

    def reset_warm_start(self) -> None:
        self._last_state_traj = None
        self._last_control_seq = None

    def solve(
        self,
        initial_state: Sequence[float],
        reference: Optional[np.ndarray] = None,
        *,
        reset_nominal: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray | float | str | bool]]:
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (self.state_dim,):
            raise ValueError(f"initial_state must have shape ({self.state_dim},)")

        if reference is not None:
            self.update_reference(reference)
        elif self._ref_positions is None or self._ref_velocities is None:
            raise ValueError("Reference trajectory has not been set")

        controlled_block = state[self._agent_slice(self.controlled_agent)].copy()

        if reset_nominal:
            self.reset_warm_start()
        self._ensure_nominal(controlled_block)

        opponent_predictions = self._predict_opponent_positions(state)
        nominal_result = self._solve_nominal_mpc(controlled_block)

        if not nominal_result["feasible"]:
            info = {
                "mpc_status": nominal_result["status"],
                "mpc_feasible": False,
                "objective": float("nan"),
                "filter_status": "SKIPPED",
                "filtered": False,
                "nominal_control": nominal_result["fallback"],
                "applied_control": nominal_result["fallback"],
            }
            return nominal_result["fallback"].copy(), info

        u_nom = nominal_result["control_traj"][0]
        print(f"Nominal MPC control: {u_nom}, nominal current state: {nominal_result['state_traj'][0]}, nominal next state: {nominal_result['state_traj'][1]}")
        filter_control, filter_info = self._apply_cbf_filter(
            controlled_block,
            u_nom,
            opponent_predictions,
            nominal_result["state_traj"],
        )

        # Prepare warm start for next iteration using the nominal plan.
        self._update_warm_start(nominal_result["state_traj"], nominal_result["control_traj"])

        info: Dict[str, np.ndarray | float | str | bool] = {
            "mpc_status": nominal_result["status"],
            "mpc_feasible": True,
            "objective": nominal_result["objective"],
            "state_traj": nominal_result["state_traj"],
            "control_traj": nominal_result["control_traj"],
            "nominal_control": u_nom,
            "applied_control": filter_control,
            "filter_status": filter_info["status"],
            "filtered": filter_info["filtered"],
        }
        if "violation" in filter_info:
            info["violation"] = filter_info["violation"]
        if "constraints" in filter_info:
            info["cbf_constraints"] = filter_info["constraints"]
        if "slack" in filter_info:
            info["cbf_slack"] = filter_info["slack"]

        return filter_control, info

    def _solve_nominal_mpc(self, controlled_block: np.ndarray) -> Dict[str, object]:
        horizon = self.config.horizon
        x_var = cp.Variable((horizon + 1, self.per_agent_state_dim))
        u_var = cp.Variable((horizon, self.control_dim))

        constraints = [x_var[0, :] == controlled_block]
        cost_terms = []

        for t in range(horizon):
            constraints.append(x_var[t + 1, :] == self.A @ x_var[t, :] + self.B @ u_var[t, :])
            constraints.append(cp.abs(u_var[t, :]) <= self.config.max_control)

            pos = x_var[t, self.position_indices]
            vel = x_var[t, self.velocity_indices]
            pos_ref = self._ref_positions[t]
            vel_ref = self._ref_velocities[t]

            cost_terms.append(cp.quad_form(pos - pos_ref, self.Qp))
            cost_terms.append(cp.quad_form(vel - vel_ref, self.Rv))
            cost_terms.append(cp.quad_form(u_var[t, :], self.Ru))

        pos_T = x_var[horizon, self.position_indices]
        vel_T = x_var[horizon, self.velocity_indices]
        cost_terms.append(cp.quad_form(pos_T - self._ref_positions[-1], self.Qp_T))
        cost_terms.append(cp.quad_form(vel_T - self._ref_velocities[-1], self.Rv_T))

        if self._last_state_traj is not None:
            x_var.value = self._last_state_traj.copy()
        if self._last_control_seq is not None:
            u_var.value = self._last_control_seq.copy()

        problem = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)
        solve_kwargs = dict(warm_start=True, **self._solver_options)

        try:
            problem.solve(solver=self._solver, **solve_kwargs)
        except cp.error.SolverError:
            problem.solve(solver=cp.ECOS, warm_start=True)

        status = problem.status
        optimal_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        feasible = status in optimal_statuses

        if not feasible:
            fallback = (
                self._last_control_seq[0].copy()
                if self._last_control_seq is not None
                else np.zeros(self.control_dim, dtype=np.float64)
            )
            return {
                "feasible": False,
                "status": status,
                "fallback": np.clip(fallback, -self.config.max_control, self.config.max_control),
            }

        x_sol = np.asarray(x_var.value, dtype=np.float64)
        u_sol = np.asarray(u_var.value, dtype=np.float64)
        objective = float(problem.value) if problem.value is not None else float("nan")

        return {
            "feasible": True,
            "status": status,
            "state_traj": x_sol,
            "control_traj": u_sol,
            "objective": objective,
        }

    def _apply_cbf_filter(
        self,
        controlled_block: np.ndarray,
        u_nominal: np.ndarray,
        opponent_predictions: Dict[int, np.ndarray],
        nominal_states: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if not self.other_agent_indices:
            return u_nominal.copy(), {"status": "NO_CONSTRAINT", "filtered": False}
        if nominal_states.shape[0] < 2:
            return u_nominal.copy(), {"status": "INSUFFICIENT_TRAJ", "filtered": False}

        radius_sq = self.config.safety_radius ** 2
        print(f"Applying CBF filter with safety radius: {self.config.safety_radius}, radius_sq: {radius_sq}")
        gamma = self.config.gamma_cbf

        u_var = cp.Variable(self.control_dim)
        slack = cp.Variable(nonneg=True)
        objective = 0.5 * cp.quad_form(u_var - u_nominal, self.filter_weight) + 0.5 * self.filter_slack_weight * cp.square(slack)
        constraints = [cp.abs(u_var) <= self.config.max_control, slack >= 0]

        next_state_expr = self.A @ controlled_block + self.B @ u_var
        p_next_expr = next_state_expr[self.position_indices]
        anchor_pos_next = nominal_states[1, self.position_indices]
        anchor_pos_cur = nominal_states[0, self.position_indices]

        print(f"anchor_pos_cur: {anchor_pos_cur}, anchor_pos_next: {anchor_pos_next}")

        active_constraints = 0
        min_violation = float("inf")

        nominal_next = self.A @ controlled_block + self.B @ u_nominal
        # print(f"Nominal next position: {nominal_next}, nominal control: {u_nominal}  ")
        constraint_details = []

        for agent_index in self.other_agent_indices:
            predicted_pos = opponent_predictions.get(agent_index)
            # print(f"Predicted positions for agent {agent_index}: {predicted_pos}")
            if predicted_pos is None or predicted_pos.shape[0] < 2:
                continue

            obs_pos_cur = predicted_pos[0]
            obs_pos_next = predicted_pos[1]

            diff_next = anchor_pos_next - obs_pos_next
            grad = 2.0 * diff_next
            if np.linalg.norm(grad) < 1e-6:
                grad = 2.0 * (diff_next + 1e-3 * np.array([1.0, 0.0, 0.0]))

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
            print(f"distance: {np.linalg.norm(anchor_pos_cur - obs_pos_cur)}")
            print(f"h_current: {h_current:.4f}, h_bar: {h_bar:.4f}, lower_bound: {lower_bound:.4f}")

        if active_constraints == 0:
            return u_nominal.copy(), {"status": "NO_CONSTRAINT", "filtered": False, "constraints": [], "slack": 0.0}

        problem = cp.Problem(cp.Minimize(objective), constraints)
        solve_kwargs = dict(warm_start=True, **self._solver_options)

        try:
            print(f"Solving CBF filter with {active_constraints} constraints, min_violation={min_violation:.4f}")
            problem.solve(solver=self._solver, **solve_kwargs)
        except cp.error.SolverError:
            problem.solve(solver=cp.ECOS, warm_start=True)

        status = problem.status
        optimal_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        feasible = status in optimal_statuses

        if not feasible or u_var.value is None:
            return np.clip(u_nominal, -self.config.max_control, self.config.max_control), {
                "status": status,
                "filtered": False,
                "violation": min_violation,
                "constraints": constraint_details,
                "slack": float(slack.value) if slack.value is not None else float("nan"),
            }

        filtered_control = np.asarray(u_var.value, dtype=np.float64)
        filtered_control = np.clip(filtered_control, -self.config.max_control, self.config.max_control)
        filtered = not np.allclose(filtered_control, u_nominal, atol=1e-4)
        print(f"CBF filter status: {status}, filtered: {filtered}, control change norm: {np.linalg.norm(filtered_control - u_nominal):.4f}")
        # printing for debugging
        # print(f"distance: {np.linalg.norm(anchor_pos_cur - obs_pos_cur)}")
        print(f"radius: {np.sqrt(radius_sq)}")
        # print(f"h_current: {h_current:.4f}, h_bar: {h_bar:.4f}, lower_bound: {lower_bound:.4f}")
        print(f"nominal linearized: {nominal_linearized:.4f}")
        print(f"slack weight: {self.filter_slack_weight}, slack value: {slack.value:.4f}")
        print(f"filtered control: {filtered_control}, nominal control: {u_nominal}, difference: {filtered_control - u_nominal}")
       
        return filtered_control, {
            "status": status,
            "filtered": filtered,
            "violation": min_violation,
            "constraints": constraint_details,
            "slack": float(slack.value) if slack.value is not None else float("nan"),
        }

    def _agent_slice(self, agent_index: int) -> slice:
        base = agent_index * self.per_agent_state_dim
        return slice(base, base + self.per_agent_state_dim)

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

    def _predict_opponent_positions(self, state: np.ndarray) -> Dict[int, np.ndarray]:
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


__all__ = ["DroneMPCConfig", "DroneMPCCBFController"]

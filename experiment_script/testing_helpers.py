"""Drone racing MPPI simulation with reachability-aware constraints."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from mppi_mpc_controller import (
    DroneMPPIConfig,
    DroneMPPIController,
    ReachabilityValueFunction,
)
from LCRL.data import Batch

@dataclass
class DroneRaceConfig:
    """Configuration for the drone racing simulation."""

    radius: float = 1.0
    target_speed: float = 0.4
    ego_speed_scale: float = 1.5
    duration: float = 0.8 #8 (ebonye)
    save_path: pathlib.Path = pathlib.Path("experiment_script/data/drone_race_mppi.npz")
    value_path: Optional[pathlib.Path] = None


class DroneRaceSimulation:
    """Simulate a drone following a quarter-circle track with MPPI control."""

    def __init__(
        self,
        sim_cfg: DroneRaceConfig,
        ctrl_cfg: DroneMPPIConfig,
        reference: Optional[str] = None,
        initial_state: Optional[np.ndarray] = None,
        reachability_value_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.ctrl_cfg = ctrl_cfg

        value_fn = (
            ReachabilityValueFunction.from_policy_path(
                str(reachability_value_path),
                device="cpu",
            )
            if reachability_value_path is not None
            else None
        )

        self.controller = DroneMPPIController(ctrl_cfg, value_function=value_fn)

        self.dt = ctrl_cfg.dt
        self.steps = int(np.ceil(sim_cfg.duration / self.dt))
        if reference == "learned_policy" and value_fn is not None and initial_state is not None:
            print(f"learned policy reference generation")
            self.reference = generate_learned_policy_reference(value_fn, initial_state, self.steps + ctrl_cfg.horizon + 5)
            print(f"reference shape: {self.reference.shape}")
            # import pdb; pdb.set_trace()
        else:
            self.reference = generate_quarter_circle_reference(
                sim_cfg.radius,
                self.steps + ctrl_cfg.horizon + 5,
                sim_cfg.target_speed * sim_cfg.ego_speed_scale,
                self.dt,
            )
        

        # Cap y to remain within [ -1, 1 ] corridor and avoid triggering abort.
        # self.reference[:, 2] = np.clip(self.reference[:, 2], -1.0, 2.0) ## ebonye 1/26/2026

        self.state = np.zeros(self.controller.state_dim, dtype=np.float64)
        if initial_state is not None:
            self.state = initial_state
        else:
            self._initialise_state()

        self.time_log = np.arange(self.steps, dtype=np.float64) * self.dt
        self.state_log = np.zeros((self.steps, self.controller.state_dim), dtype=np.float64)
        self.control_log = np.zeros((self.steps, self.controller.control_dim), dtype=np.float64)
        self.disturbance_log = np.zeros((self.steps, self.controller.disturbance_dim), dtype=np.float64)
        self.reference_log = np.zeros((self.steps, self.controller.per_agent_state_dim), dtype=np.float64)
        self.safety_margin_log = np.zeros(self.steps, dtype=np.float64)
        self.value_log = np.zeros(self.steps, dtype=np.float64)

    def _initialise_state(self) -> None:
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
        other_states: Sequence[np.ndarray] = [
            state_on_arc(theta, z) for theta, z in zip(other_thetas, other_z)
        ]

        other_iter = iter(other_states)
        fallback = other_states[-1] if other_states else controlled_state
        for idx in range(self.controller.num_agents):
            if idx == self.controller.controlled_agent:
                state_vec = controlled_state
            else:
                state_vec = next(other_iter, fallback)

            # Add small random perturbations so each episode starts from a slightly different pose.
            pos_noise = self.controller.rng.normal(0.0, 0.2, size=3)
            vel_noise = self.controller.rng.normal(0.0, 0.0, size=3)
            # noise = np.array(
            #     [pos_noise[0], vel_noise[0], pos_noise[1], vel_noise[1], pos_noise[2], vel_noise[2]],
            #     dtype=np.float64,
            # )
            # state_vec = state_vec + noise

            sl = slice(
                idx * self.controller.per_agent_state_dim,
                (idx + 1) * self.controller.per_agent_state_dim,
            )
            self.state[sl] = state_vec

    def run(self) -> Dict[str, np.ndarray]:
        self.controller.reset_sequence()
        num_ref = self.reference.shape[0]
        final_steps = self.steps

        for t in range(self.steps):
            ref_start = min(t, num_ref - self.ctrl_cfg.horizon - 1)
            ref_segment = self.reference[ref_start : ref_start + self.ctrl_cfg.horizon]

            control, info = self.controller.solve(
                self.state,
                ref_segment,
                reset_nominal=(t == 0),
            )

            self.state_log[t] = self.state
            self.control_log[t] = control
            disturbance = info.get(
                "best_disturbances",
                np.zeros((self.ctrl_cfg.horizon, self.controller.disturbance_dim)),
            )
            value_traj = info.get("best_values")
            self.disturbance_log[t] = disturbance[0]
            # print(f"ref_segment[0]: {ref_segment[0]}")
            self.reference_log[t] = ref_segment[0]
            self.safety_margin_log[t] = float(info["min_safety_margin"])
            self.value_log[t] = (
                value_traj[0] if value_traj is not None and value_traj.size > 0 else np.nan
            )

            if info.get("planning_aborted", False):
                final_steps = t + 1
                break

            next_state = self.controller.simulate_step(self.state, control, disturbance[0])
            self.state = next_state

        return {
            "time": self.time_log[:final_steps].copy(),
            "states": self.state_log[:final_steps].copy(),
            "controls": self.control_log[:final_steps].copy(),
            "disturbances": self.disturbance_log[:final_steps].copy(),
            "references": self.reference_log[:final_steps].copy(),
            "safety_margin": self.safety_margin_log[:final_steps].copy(),
            "value_estimate": self.value_log[:final_steps].copy(),
        }

    @staticmethod
    def save(path: pathlib.Path, data: Dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)


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

def generate_learned_policy_reference(
    policy_function: ReachabilityValueFunction,
    start_state: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """Generate reference states by rolling out a learned policy from a start state."""

    def find_a(state):
        tmp_obs = np.array(state).reshape(1,-1)
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = policy_function.policy(tmp_batch, model = "actor_old").act
        act = policy_function.policy.map_action(tmp).cpu().detach().numpy().flatten()
        return act


    state_dim = start_state.shape[0]
    per_agent_state_dim = policy_function.per_agent_state_dim
    if state_dim % per_agent_state_dim != 0:
        raise ValueError("start_state has incompatible dimension with policy_function.")
    num_agents = state_dim // per_agent_state_dim
    reference = np.zeros((num_points, per_agent_state_dim), dtype=np.float64)
    current_state = start_state.copy()
    cnfg = DroneMPPIConfig(num_agents=1)
    cntrllr = DroneMPPIController(cnfg, value_function=policy_function)  
    # print(f"num_agents = {cntrllr.num_agents}")  
    reference[0] = current_state[:per_agent_state_dim]

    print(f"num_points: {num_points}")
    from tqdm import tqdm
    # for t in range(num_points - 1):
    for t in tqdm(tqdm(range(num_points-1))):
        # controlled_sl = slice(
        #     policy_function.controlled_agent * per_agent_state_dim,
        #     (policy_function.controlled_agent + 1) * per_agent_state_dim,
        # )
        # reference[t] = current_state[controlled_sl]

        action = find_a(current_state)[:3]  # extract control for controlled agent
        next_state = cntrllr.simulate_step(current_state, action)
        current_state = next_state
        reference[t+1] = current_state[:per_agent_state_dim]
    return reference


 
    


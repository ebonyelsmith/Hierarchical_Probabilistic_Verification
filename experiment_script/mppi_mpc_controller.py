"""MPPI controller for drone racing with reachability constraints.

This module provides:

* ``ReachabilityValueFunction``: utility for loading a trained DDPG reach-avoid
  policy and evaluating its critic as a value function surrogate.
* ``DroneMPPIController``: a sampling-based MPC (MPPI) controller that follows
  the ``Three_Drones`` dynamics and enforces both distance-based and learned
  reachability constraints. The controller can treat any agent as the
  controlled entity and reorders state vectors to share a symmetric viability
  kernel across agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from LCRL.policy.modelfree.ddpg_reach_avoid_game_new import (
    reach_avoid_game_DDPGPolicy,
)
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic

EPS = 1e-9


def _infer_hidden_sizes(state_dict: Dict[str, torch.Tensor], prefix: str) -> Sequence[int]:
    """Infer hidden layer sizes from a stored policy state dict."""
    hidden = []
    idx = 0
    while True:
        key = f"{prefix}preprocess.model.model.{idx}.weight"
        if key not in state_dict:
            break
        hidden.append(state_dict[key].shape[0])
        idx += 2  # linear layers are stored at even indices; activation has no weights
    return hidden


class ReachabilityValueFunction:
    """Wrapper around a trained reach-avoid DDPG policy for value evaluation."""

    per_agent_state_dim: int = 6

    def __init__(
        self,
        policy: reach_avoid_game_DDPGPolicy,
        device: torch.device,
    ) -> None:
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device

        self.state_dim = policy.actor1.preprocess.model.model[0].weight.shape[1]
        self.control_dim = policy.actor1.last.model[0].weight.shape[0]
        self.disturbance_dim = policy.actor2.last.model[0].weight.shape[0]

        self.num_agents = 1 + self.disturbance_dim // self.control_dim
        if self.state_dim != self.num_agents * self.per_agent_state_dim:
            raise ValueError(
                "State dimension mismatch when loading reachability value function."
            )

    def _rotated_agent_order(self, agent_index: int) -> Sequence[int]:
        """Return agent indices ordered relative to ``agent_index``."""
        return [
            (agent_index + 1 + offset) % self.num_agents
            for offset in range(self.num_agents - 1)
        ]

    @classmethod
    def from_policy_path(
        cls,
        policy_path: str,
        device: str | torch.device = "cpu",
    ) -> "ReachabilityValueFunction":
        """Instantiate from a saved ``policy.pth`` produced by run_training_ddpg."""

        policy_state = torch.load(policy_path, map_location=device)

        actor1_hidden = _infer_hidden_sizes(policy_state, "actor1.")
        actor2_hidden = _infer_hidden_sizes(policy_state, "actor2.")
        critic_hidden = _infer_hidden_sizes(policy_state, "critic.")

        state_dim = policy_state["actor1.preprocess.model.model.0.weight"].shape[1]
        control_dim = policy_state["actor1.last.model.0.weight"].shape[0]
        disturbance_dim = policy_state["actor2.last.model.0.weight"].shape[0]
        total_action_dim = control_dim + disturbance_dim

        activation = nn.ReLU
        device_t = torch.device(device)

        critic_net = Net(
            state_dim,
            total_action_dim,
            hidden_sizes=critic_hidden,
            activation=activation,
            concat=True,
            device=device_t,
        )
        critic = Critic(critic_net, device=device_t).to(device_t)
        critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

        actor1_net = Net(
            state_dim,
            hidden_sizes=actor1_hidden,
            activation=activation,
            device=device_t,
        )
        actor1 = Actor(
            actor1_net, control_dim, max_action=1.0, device=device_t
        ).to(device_t)
        actor1_optim = optim.Adam(actor1.parameters(), lr=1e-4)

        actor2_net = Net(
            state_dim,
            hidden_sizes=actor2_hidden,
            activation=activation,
            device=device_t,
        )
        actor2 = Actor(
            actor2_net, disturbance_dim, max_action=1.0, device=device_t
        ).to(device_t)
        actor2_optim = optim.Adam(actor2.parameters(), lr=1e-4)

        policy = reach_avoid_game_DDPGPolicy(
            critic=critic,
            critic_optim=critic_optim,
            actor1=actor1,
            actor1_optim=actor1_optim,
            actor2=actor2,
            actor2_optim=actor2_optim,
            tau=0.005,
            # gamma=0.99,
            gamma=0.95,
            actor_gradient_steps=1,
        )
        policy.load_state_dict(policy_state)

        return cls(policy, device_t)

    def _format_state(
        self,
        state: np.ndarray,
        agent_index: int,
        state_perm: np.ndarray,
    ) -> np.ndarray:
        blocks = [
            state[i * self.per_agent_state_dim : (i + 1) * self.per_agent_state_dim]
            for i in range(self.num_agents)
        ]
        rotated = blocks[agent_index:] + blocks[:agent_index]
        permuted = [block[state_perm] for block in rotated]
        return np.concatenate(permuted)

    def _map_disturbance(
        self,
        disturbance: torch.Tensor,
        agent_index: int,
    ) -> np.ndarray:
        """Convert actor2 output into environment ordering."""
        per_agent = (
            disturbance.reshape(self.num_agents - 1, self.control_dim)
            .cpu()
            .numpy()
        )

        env_order = np.zeros((self.num_agents, self.control_dim), dtype=np.float64)
        for offset, mapped_idx in enumerate(self._rotated_agent_order(agent_index)):
            env_order[mapped_idx] = per_agent[offset]

        env_order = np.delete(env_order, agent_index, axis=0)
        return env_order.reshape(-1)

    @torch.no_grad()
    def evaluate(
        self,
        state: np.ndarray,
        control: np.ndarray,
        agent_index: int = 0,
        state_perm: Optional[Sequence[int]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Return value estimate and the associated disturbance action.

        ``agent_index`` selects which agent is regarded as the controller; the state
        is rotated accordingly so that the reachability network can be reused in a
        symmetric fashion for all agents. ``state_perm`` optionally permutes the
        per-agent state layout before evaluation (e.g. to map between
        [x, vx, y, vy, z, vz] and [x, y, z, vx, vy, vz]).
        """

        perm = (
            np.asarray(state_perm, dtype=np.int64)
            if state_perm is not None
            else np.arange(self.per_agent_state_dim, dtype=np.int64)
        )

        formatted_state = self._format_state(
            np.asarray(state, dtype=np.float64), agent_index, perm
        )

        state_t = torch.as_tensor(
            formatted_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        control_t = torch.as_tensor(
            np.clip(control, -1.0, 1.0), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        disturbance_t, _ = self.policy.actor2(state_t)
        combined_action = torch.cat([control_t, disturbance_t], dim=1)
        value_t = self.policy.critic(state_t, combined_action)
        

        disturbance_out = self._map_disturbance(disturbance_t, agent_index)

        return float(value_t.item()), disturbance_out


@dataclass
class DroneMPPIConfig:
    """Configuration container for the drone MPPI controller."""

    horizon: int = 5 #25 # Ebonye 1/27/2026
    dt: float = 0.1
    num_samples: int = 1000
    temperature: float = 1.0
    noise_sigma: Sequence[float] = (0.4, 0.4, 0.4)
    position_weight: float = 5.0
    velocity_weight: float = 2.0
    control_weight: float = 0.1
    safety_radius: float = 0
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


class DroneMPPIController:
    """MPPI controller with reachability-aware penalties."""

    def __init__(
        self,
        config: DroneMPPIConfig,
        value_function: Optional[ReachabilityValueFunction] = None,
    ) -> None:
        self.config = config
        self.value_function = value_function

        self.num_agents = config.num_agents
        self.per_agent_state_dim = config.per_agent_state_dim
        self.control_dim = config.per_agent_control_dim
        self.state_dim = self.num_agents * self.per_agent_state_dim
        self.disturbance_dim = (self.num_agents - 1) * self.control_dim
        self.controlled_agent = config.controlled_agent_index
        self.other_agent_indices = [
            idx for idx in range(self.num_agents) if idx != self.controlled_agent
        ]
        self.state_perm = np.asarray(config.viability_state_perm, dtype=np.int64)

        self.nominal_sequence = np.zeros(
            (config.horizon, self.control_dim), dtype=np.float64
        )
        self.rng = np.random.default_rng(config.seed)

        self._ref_positions = np.zeros((config.horizon, 3), dtype=np.float64)
        self._ref_velocities = np.zeros((config.horizon, 3), dtype=np.float64)

        # Feedback terms for opponent drones, shared with the environment
        self._K1 = np.array([3.1127], dtype=np.float64)
        # self._K2 = np.array([19.1704, 16.8205], dtype=np.float64) ## OLD (Ebonye 1/26/2026)
        self._K2 = np.array([9.1704, 16.8205], dtype=np.float64)
        self._x_star = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0], dtype=np.float64)

    def reset_sequence(self) -> None:
        self.nominal_sequence.fill(0.0)

    def _agent_slice(self, agent_index: int) -> slice:
        base = agent_index * self.per_agent_state_dim
        return slice(base, base + self.per_agent_state_dim)

    def _agent_position(self, state: np.ndarray, agent_index: int) -> np.ndarray:
        block = state[self._agent_slice(agent_index)]
        return np.array([block[0], block[2], block[4]], dtype=np.float64)

    def _agent_velocity(self, state: np.ndarray, agent_index: int) -> np.ndarray:
        block = state[self._agent_slice(agent_index)]
        return np.array([block[1], block[3], block[5]], dtype=np.float64)

    def _set_reference(self, reference: np.ndarray) -> None:
        if reference.ndim != 2:
            raise ValueError("reference must be a 2-D array.")
        if reference.shape[1] not in (3, self.per_agent_state_dim):
            raise ValueError("reference must contain 3 (pos) or full state columns.")

        trimmed = reference[: self.config.horizon]
        if trimmed.shape[0] == 0:
            raise ValueError("reference must contain at least one waypoint.")

        positions = np.zeros((trimmed.shape[0], 3), dtype=np.float64)
        positions[:, 0] = trimmed[:, 0]
        positions[:, 1] = trimmed[:, 2]
        positions[:, 2] = trimmed[:, 4] if trimmed.shape[1] >= 5 else 0.0

        velocities = np.zeros_like(positions)
        if trimmed.shape[1] >= self.per_agent_state_dim:
            velocities[:, 0] = trimmed[:, 1]
            velocities[:, 1] = trimmed[:, 3]
            velocities[:, 2] = trimmed[:, 5]

        if trimmed.shape[0] < self.config.horizon:
            pad_len = self.config.horizon - trimmed.shape[0]
            pos_list = [positions]
            vel_list = [velocities]
            last_pos = positions[-1].copy()
            last_vel = velocities[-1].copy()
            for _ in range(pad_len):
                last_pos = last_pos + self.config.dt * last_vel
                pos_list.append(last_pos[np.newaxis, :])
                vel_list.append(last_vel[np.newaxis, :])
            positions = np.vstack(pos_list)
            velocities = np.vstack(vel_list)

        self._ref_positions = positions
        self._ref_velocities = velocities

    def _safety_margin(self, state: np.ndarray) -> float:
        controlled_pos = self._agent_position(state, self.controlled_agent)
        margins = []
        for idx in self.other_agent_indices:
            pos = self._agent_position(state, idx)
            margins.append(np.linalg.norm(controlled_pos - pos) - self.config.safety_radius)
        return float(min(margins)) if margins else float("inf")

    def _stage_cost(
        self,
        state: np.ndarray,
        control: np.ndarray,
        t: int,
        value: Optional[float],
    ) -> Tuple[float, float]:
        pos_error = self._agent_position(state, self.controlled_agent) - self._ref_positions[t]
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

    def _evaluate_viability(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> Tuple[Optional[float], np.ndarray]:
        if self.value_function is None:
            return None, np.zeros(self.disturbance_dim, dtype=np.float64)

        value, disturbance = self.value_function.evaluate(
            state,
            control,
            agent_index=self.controlled_agent,
            state_perm=self.state_perm,
        )

        if disturbance.size == 0:
            disturbance = np.zeros(self.disturbance_dim, dtype=np.float64)
        else:
            disturbance = disturbance.astype(np.float64, copy=False)

        return value, disturbance

    def _opponent_feedback(self, state: np.ndarray, agent_index: int) -> np.ndarray:
        block = state[self._agent_slice(agent_index)]
        ax = float(
            self._K2 @ np.array([self._x_star[0] - block[0], self._x_star[1] - block[1]])
        )
        ay = float(self._K1 @ np.array([self._x_star[3] - block[3]]))
        az = float(
            self._K2 @ np.array([self._x_star[4] - block[4], self._x_star[5] - block[5]])
        )
        return np.array([ax, ay, az], dtype=np.float64)
    
    ## Ebonye 2/1/2026: setting opponent gain
    def set_opponent_gain(self, gain: float) -> None:
        self.config.opponent_gain = gain

    def simulate_step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        next_state = np.array(state, dtype=np.float64, copy=True)
        ctrl = np.clip(control, -1.0, 1.0)
        dt = self.config.dt
        cg_control = self.config.control_gain
        cg_opponent = self.config.opponent_gain
        dg = self.config.disturbance_gain

        disturbance = (
            np.zeros(self.disturbance_dim, dtype=np.float64)
            if disturbance is None
            else np.asarray(disturbance, dtype=np.float64)
        )
        disturbance_map: Dict[int, np.ndarray] = {}
        for k, agent_idx in enumerate(self.other_agent_indices):
            sl = slice(k * self.control_dim, (k + 1) * self.control_dim)
            disturbance_map[agent_idx] = disturbance[sl]

        opponent_feedbacks: Dict[int, np.ndarray] = {}  # Cache feedback computations

        for agent_idx in range(self.num_agents):
            block_slice = self._agent_slice(agent_idx)
            block = next_state[block_slice]
            orig_block = state[block_slice]

            block[0] = orig_block[0] + dt * orig_block[1]
            block[2] = orig_block[2] + dt * orig_block[3]
            block[4] = orig_block[4] + dt * orig_block[5]

            if agent_idx == self.controlled_agent:
                block[1] = orig_block[1] + dt * cg_control * ctrl[0]
                block[3] = orig_block[3] + dt * cg_control * ctrl[1]
                block[5] = orig_block[5] + dt * cg_control * ctrl[2]
            else:
                feedback = self._opponent_feedback(state, agent_idx)
                if agent_idx not in opponent_feedbacks:
                    opponent_feedbacks[agent_idx] = feedback
                disturb = disturbance_map.get(agent_idx, np.zeros(self.control_dim, dtype=np.float64))
                accel = cg_opponent * feedback + dg * disturb
                block[1] = orig_block[1] + dt * accel[0]
                block[3] = orig_block[3] + dt * accel[1]
                block[5] = orig_block[5] + dt * accel[2]

            next_state[block_slice] = block

        return next_state, opponent_feedbacks # Cache feedback computations

    def rollout(
        self,
        initial_state: np.ndarray,
        control_seq: np.ndarray,
    ) -> Tuple[
        float,
        np.ndarray,
        float,
        float,
        np.ndarray,
        np.ndarray,
    ]:
        states = np.zeros(
            (self.config.horizon + 1, self.state_dim), dtype=np.float64
        )
        states[0] = initial_state

        disturbances = np.zeros(
            (self.config.horizon, self.disturbance_dim), dtype=np.float64
        )
        values = np.full(self.config.horizon, np.inf, dtype=np.float64)

        total_cost = 0.0
        min_margin = np.inf
        min_value = np.inf

        for t in range(self.config.horizon):
            control = control_seq[t]

            value, disturbance = self._evaluate_viability(states[t], control)
            if value is not None:
                values[t] = value
                min_value = min(min_value, value)

            disturbances[t] = disturbance

            cost, margin = self._stage_cost(states[t], control, t, value)
            total_cost += cost
            min_margin = min(min_margin, margin)

            # need to set opponent gain here after doing MLE so it is changed when simulating step (Ebonye 2/2/2026)
            states[t + 1], _ = self.simulate_step(states[t], control, disturbances[t])

        return total_cost, states, float(min_margin), float(min_value), disturbances, values

    def solve(
        self,
        initial_state: Sequence[float],
        reference: np.ndarray,
        reset_nominal: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        state = np.asarray(initial_state, dtype=np.float64)
        # print(f"state shape: {state.shape}, expected: ({self.state_dim},)")
        # print(f"num_agents: {self.num_agents}")
        if state.shape != (self.state_dim,):
            raise ValueError(f"initial_state must have shape ({self.state_dim},).")

        if reset_nominal:
            self.reset_sequence()

        self._set_reference(np.asarray(reference, dtype=np.float64))

        # Stop planning if the controlled agent has already exited the corridor (y > 1).
        if self._agent_position(state, self.controlled_agent)[1] > 1.0:
            control_to_apply = np.zeros(self.control_dim, dtype=np.float64)
            info: Dict[str, np.ndarray] = {
                "planning_aborted": np.array(True),
                "reason": np.array("controlled agent y position exceeded 1.0"),
                "state": state.copy(),
                "nominal_sequence": self.nominal_sequence.copy(),
                "min_safety_margin": np.array(np.inf, dtype=np.float64),
                "min_value": np.array(np.inf, dtype=np.float64),
                "best_disturbances": np.zeros(
                    (self.config.horizon, self.disturbance_dim), dtype=np.float64
                ),
                "best_values": np.zeros(self.config.horizon, dtype=np.float64),
            }
            return control_to_apply, info

        ns = self.config.num_samples
        horizon = self.config.horizon

        control_samples = np.zeros((ns, horizon, self.control_dim), dtype=np.float64)
        noise_samples = np.zeros_like(control_samples)
        traj_samples = np.zeros((ns, horizon + 1, self.state_dim), dtype=np.float64)
        cost_samples = np.zeros(ns, dtype=np.float64)
        margin_samples = np.full(ns, np.inf, dtype=np.float64)
        value_samples = np.full(ns, np.inf, dtype=np.float64)
        disturbance_samples = np.zeros(
            (ns, horizon, self.disturbance_dim), dtype=np.float64
        )
        value_traj_samples = np.full((ns, horizon), np.inf, dtype=np.float64)

        for i in range(ns):
            noise = (
                self.rng.standard_normal((horizon, self.control_dim))
                * self.config.noise_sigma
            )
            controls = np.clip(self.nominal_sequence + noise, -1.0, 1.0)
            noise = controls - self.nominal_sequence

            (
                total_cost,
                states,
                min_margin,
                min_value,
                disturbances,
                values,
            ) = self.rollout(state, controls)

            control_samples[i] = controls
            noise_samples[i] = noise
            traj_samples[i] = states
            cost_samples[i] = total_cost
            margin_samples[i] = min_margin
            value_samples[i] = min_value
            disturbance_samples[i] = disturbances
            value_traj_samples[i] = values

        beta = float(cost_samples.min())
        weights = np.exp(-(cost_samples - beta) / self.config.temperature)
        weight_sum = float(weights.sum()) + EPS

        weighted_noise = np.einsum("i,ijk->jk", weights, noise_samples) / weight_sum
        self.nominal_sequence = np.clip(
            self.nominal_sequence + weighted_noise,
            -1.0,
            1.0,
        )

        best_idx = int(np.argmin(cost_samples))
        best_traj = traj_samples[best_idx]
        best_controls = control_samples[best_idx]
        best_disturbances = disturbance_samples[best_idx]
        best_values = value_traj_samples[best_idx]

        control_to_apply = self.nominal_sequence[0].copy()
        info: Dict[str, np.ndarray] = {
            "costs": cost_samples,
            "weights": weights,
            "best_traj": best_traj,
            "best_controls": best_controls,
            "best_disturbances": best_disturbances,
            "best_values": best_values,
            "nominal_sequence": self.nominal_sequence.copy(),
            "min_safety_margin": np.array(margin_samples[best_idx], dtype=np.float64),
            "min_value": np.array(value_samples[best_idx], dtype=np.float64),
        }

        self.nominal_sequence[:-1] = self.nominal_sequence[1:]
        self.nominal_sequence[-1] = 0.0

        return control_to_apply, info


__all__ = ["DroneMPPIConfig", "DroneMPPIController", "ReachabilityValueFunction"]

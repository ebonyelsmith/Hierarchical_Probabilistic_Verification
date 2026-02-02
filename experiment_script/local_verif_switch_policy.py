import argparse
import os
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
from scipy.interpolate import RegularGridInterpolator

from local_verif_utils import get_beta5, beta, calibrate_V_vectorized, calibrate_V_scenario_local_vectorized, grow_regions_closest_point, make_new_env, compute_min_scenarios_alex


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
        print(f"Verified value function at state {state}: {value}")
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
        policy_function: ReachabilityValueFunction,
        confidence: float = 0.97,
        delta: float = 1e-3) -> VerifiedReachableSet:

        """Compute the verified reachable set using sampling and Lipschitz constants."""

        if self.alphaC_scenario_list is None or self.alphaC_list is None:
            self.alphaC_scenario_list, self.alphaR_scenario_list, self.alphaC_list, self.alphaR_list = np.zeros(self.reachability_horizon), np.zeros(self.reachability_horizon), np.zeros(self.reachability_horizon), np.zeros(self.reachability_horizon)
            env = make_new_env(args)
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
        mppi_cfg: DroneMPPIConfig,
        policy_function: ReachabilityValueFunction,
        verified_reachable_set: VerifiedReachableSet,
    ):
        self.args = args
        self.mppi_cfg = mppi_cfg
        # self.mppi_controller = DroneMPPIController(self.mppi_cfg)
        self.mppi_controller_local = DroneMPPIControllerLocalVerif(self.mppi_cfg, value_function=policy_function)
        self.mppi_controller_fast = DroneMPPIController(self.mppi_cfg, value_function=policy_function)
        self.policy_function = policy_function.policy
        self.verified_reachable_set = verified_reachable_set
        # self.target_set_reached = False
        self.Lf = 1.05125 
        self.Lc = 20
        self.Lr = 10
        self.epsilon_x = 0.1
        self.reachability_horizon = 30  # steps
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
            next_state = cntrllr.simulate_step(current_state, action)
            current_state = next_state
            reference[t+1] = current_state[:per_agent_state_dim]
        return reference


    def solve(
        self,
        state: np.ndarray,
        reset_nominal: bool = False,
    ) -> np.ndarray:
        """Return control action for the ego drone given the current state."""

        # Evaluate the verified value function at current state to decide if safe
        ego_xy = state[[0, 2]]
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

        mode = 0 # 0: learned policy, 1: local verif + mppi, 2: high-speed lane maintain

        if not is_in_target_set:
            if self.recompute:
                self.verified_reachable_set_computer.current_state = state
                # print(f"self.policy_function: {self.policy_function}")
                verified_set_deterministic, _ = self.verified_reachable_set_computer.compute_verified_set(
                    self.args,
                    self.policy_function,   
                )
                self.verified_reachable_set = verified_set_deterministic
                
                # import pdb; pdb.set_trace()  # update to new verified set
        
            is_safe = self.verified_reachable_set.is_inside(ego_xy)
            # print(f"is_safe: {is_safe}")
            # import pdb; pdb.set_trace()

            if is_safe:
                # Inside verified BRS (given new intent): use learned policy
                action = self.find_a(state)
                mode = 0
                return action, mode
            else:
                env = make_new_env(self.args)

                if self.recompute_local:
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
                        self.verified_reachable_set.value_function,
                        X,
                        Y,
                        env,
                        self.reachability_horizon,
                        self.verified_reachable_set_computer.alphaC_list,
                        self.verified_reachable_set_computer.alphaR_list,
                        self.policy_function,
                        self.args,
                        max_attept_radius = 1.0, #0.5,
                        N_samples = n_samples,
                        tol=1e-2
                    )
                    self.expanded_region = expanded_region
                    x_center, y_center, r_safe = expanded_region
                    # check if current state is inside local growth set
                    is_safe_local = np.linalg.norm(ego_xy - np.array([x_center, y_center])) <= r_safe
                    self.is_safe_local = is_safe_local
                    self.recompute_local = False  # reset flag after recomputing local growth set

                # if is_safe_local:
                if self.is_safe_local:
                    # inside local growth set: use learned policy
                    action = self.find_a(state)
                    mode = 0
                    return action, mode
                else:
                    x_center, y_center, r_safe = self.expanded_region
                    center = np.array([x_center, y_center])
                    direction = ego_xy - center
                    direction_norm = np.linalg.norm(direction)

                    closest_safe_point = center + (r_safe / direction_norm) * direction
                    closest_safe_point = np.array([closest_safe_point[0], closest_safe_point[1], state[4]])  # keep z coordinate the same

                    self.mppi_controller_local.closest_safe_point = closest_safe_point
                    reference = self.generate_learned_policy_reference(
                        self.policy_function,
                        state,
                        self.mppi_cfg.horizon
                    )
                    action, _ = self.mppi_controller_local.solve(state, reference, reset_nominal)
                    mode = 1
                    return action, mode
        
        else:
            # Safely ahead of opponent: maintain high speed along lane center using MPPI
            desired_speed = 1.5  # m/s
            desired_position = np.array([0.0, state[1], state[2]])  # keep current y, z positions
            desired_velocity = np.array([0.0, desired_speed, 0.0])  # high speed along y-axis
            desired_state = np.array([desired_position[0], desired_velocity[0],
                             desired_position[1], desired_velocity[1],
                             desired_position[2], desired_velocity[2]])
            reference = np.tile(desired_state, (self.mppi_cfg.horizon, 1))
            # ref_traj = np.tile(desired_position, (self.mppi_cfg.horizon, 1))
            # ref_velocities = np.tile(np.array([desired_speed, 0.0, 0.0]), (self.mppi_cfg.horizon, 1))
            # self.mppi_controller_fast._set_reference(ref_traj, ref_velocities)
            # action = self.mppi_controller_fast.solve(state, reset_nominal)
            action, _ = self.mppi_controller_fast.solve(state, reference, reset_nominal)
            mode = 2
            return action, mode
        

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
            mppi_cfg=mppi_cfg,
            policy_function=value_fn,
            verified_reachable_set=offline_verified_set,            
        )
        self.state = initial_state
        self.state_log = np.zeros((self.num_steps, initial_state.shape[0]), dtype=np.float64)
        self.control_log = np.zeros((self.num_steps, mppi_cfg.per_agent_control_dim), dtype=np.float64)

    def run(self) -> Dict[str, np.ndarray]:
        """Run the drone racing simulation."""

        for t in range(self.num_steps):
            reset_nominal = t == 0
            action, mode = self.controller.solve(self.state, reset_nominal)
            print(f"Step {t}, State: {self.state}, Action: {action}, Mode: {mode}")
            self.state_log[t] = self.state
            self.control_log[t] = action[:3]
            self.state = self.controller.mppi_controller_fast.simulate_step(self.state, action)

        return {
            "state_log": self.state_log,
            "control_log": self.control_log,
        }
    
def extract_xy(states: np.ndarray, agent_index: int, per_agent_state_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) position history for a given agent."""

    sl = slice(agent_index * per_agent_state_dim, (agent_index + 1) * per_agent_state_dim)
    agent_states = states[:, sl]
    return agent_states[:, 0], agent_states[:, 2]


        
def plot_trajectories(
    states: np.ndarray,
    sim: DroneRaceSimulation,
    save_path: pathlib.Path | None = None,
    gif_path: pathlib.Path | None = None,
    fps: int = 20,
) -> None:
    """Plot ego/opponent xy-trajectories and the quarter-circle reference."""

    controller = sim.controller
    # per_agent_state_dim = controller.per_agent_state_dim
    per_agent_state_dim = controller.mppi_controller_fast.per_agent_state_dim

    fig, ax = plt.subplots(figsize=(6, 6))

    # vel magnitude per agent
    max_speeds = []
    for idx in range(controller.mppi_controller_fast.num_agents):
        sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
        vel = states[:, sl][:, [1, 3, 5]]
        speed = np.linalg.norm(vel, axis=1)
        max_speeds.append(speed.max())

    # # Reference quarter-circle
    # ref_xy = sim.reference[:, ::2][:, :2]
    # print(f"Ref xy shape: {ref_xy.shape}")
    # ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = []
    for idx in range(controller.mppi_controller_fast.num_agents):
        x, y = extract_xy(states, idx, per_agent_state_dim)
        lbl = "MPPI agent" if idx == controller.mppi_controller_fast.controlled_agent else f"Agent {idx}"
        labels.append(ax.plot(x, y, color=colours[idx % len(colours)], label=lbl)[0])

    ax.scatter(0.0, 0.0, marker="s", color="black", label="Gate (origin)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("MPPI Drone Trajectory")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    print("Maximum speeds (m/s) per agent:")
    for idx, ms in enumerate(max_speeds):
        tag = "MPPI agent" if idx == controller.mppi_controller_fast.controlled_agent else f"Agent {idx}"
        print(f"  {tag}: {ms:.3f}")

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
        for idx in range(controller.mppi_controller_fast.num_agents):
            x, y = extract_xy(states, idx, per_agent_state_dim)
            sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
            agent_states = states[:, sl]
            vx = agent_states[:, 1]
            vy = agent_states[:, 3]
            vz = agent_states[:, 5]
            trajectories.append((x, y))
            velocities.append((vx, vy))
            speed_components.append((vx, vy, vz))

        lines = []
        quivers = []
        speed_texts = []
        for idx in range(controller.mppi_controller_fast.num_agents):
            colour = colours[idx % len(colours)]
            lbl = "MPPI agent" if idx == controller.mppi_controller_fast.controlled_agent else f"Agent {idx}"
            line, = ax_gif.plot([], [], color=colour, label=lbl)
            marker, = ax_gif.plot([], [], "o", color=colour)
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
            artists = []
            for (
                (line, marker),
                quiver,
                text,
                (x_hist, y_hist),
                (vx_hist, vy_hist),
                (vx_comp, vy_comp, vz_comp),
            ) in zip(
                lines, quivers, speed_texts, trajectories, velocities, speed_components
            ):
                line.set_data(x_hist[: frame + 1], y_hist[: frame + 1])
                marker.set_data([x_hist[frame]], [y_hist[frame]])
                vx = vx_hist[frame] * vel_arrow_scale
                vy = vy_hist[frame] * vel_arrow_scale
                quiver.set_offsets(np.array([[x_hist[frame], y_hist[frame]]]))
                quiver.set_UVC(np.array([vx]), np.array([vy]))
                speed = np.sqrt(
                    vx_comp[frame] ** 2 + vy_comp[frame] ** 2 + vz_comp[frame] ** 2
                )
                text.set_position((x_hist[frame] + 0.05, y_hist[frame] + 0.05))
                text.set_text(f"{speed:.2f} m/s")
                artists.extend([line, marker, quiver, text])
            return artists

        from matplotlib import animation

        blit_flag = gif_path is None
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
    return parser.parse_args()




def main() -> None:
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)

    # import pdb; pdb.set_trace()

    current_state = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])
    verif_reach_set_computer = ComputingVerifiedReachableSet(
        current_state=current_state,
    )
    verified_set_deterministic, verified_set_scenario = verif_reach_set_computer.compute_verified_set(
        args,
        policy_function
    )

    # Plotting the verified reachable sets
    x = np.arange(-0.9, 0.9, verif_reach_set_computer.epsilon_x)
    y = np.arange(-2.6, 0, verif_reach_set_computer.epsilon_x)
    X, Y = np.meshgrid(x, y)
    V_det = verified_set_deterministic.value_function
    # V_scen = verified_set_scenario.value_function

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

    ax.plot_surface(X, Y, V_det, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)
    ax.set_title("Verified Reachable Set (Deterministic)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Value Function")

    # ax2.plot_surface(X, Y, V_scen, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)
    # ax2.set_title("Verified Reachable Set (Scenario)")
    # ax2.set_xlabel("X Position")
    # ax2.set_ylabel("Y Position")
    # ax2.set_zlabel("Value Function")

    # save figures
    fig.savefig("verified_reachable_set_deterministic.png")
    # fig2.savefig("verified_reachable_set_scenario.png")

def main2() -> None:
    args = get_args()
    args2 = parse_args()
    env, policy_function = get_env_and_policy(args)
    initial_state = np.array([-0.76, 0.0, -2.5, 0.7, 0.0, 0.0, 0.4, 0.0, -2.2, 0.3, 0.0, 0.0])

    race_config = DroneRaceConfig(
        duration=3.0,
        initial_state=initial_state,
        value_path=args2.value_path,
    )
    mppi_config = DroneMPPIConfig(controlled_agent_index=args2.controlled_agent)


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
    # import pdb; pdb.set_trace()
    plot_trajectories(
        states=data["state_log"],
        sim=sim,
        save_path=args2.save_figure,
        gif_path=args2.save_gif,
        fps=20,
    )
    
    


if __name__ == "__main__":
    # main()
    main2()






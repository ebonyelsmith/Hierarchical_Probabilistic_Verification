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
        self.mppi_cfg = deepcopy(mppi_cbf_cfg)
        self.mppi_cfg.mppi_cfg.horizon = 20  # increase horizon for better performance in the MPPI baseline with CBF
        self.mppi_cfg.mppi_cfg.safety_radius = 0.2
        # self.mppi_cfg.mppi_cfg.opponent_gain = 0.5  # assume a fixed opponent gain for the MPPI baseline
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

        # import pdb; pdb.set_trace()
        for t in range(self.num_steps):
        # for t in tqdm(range(self.num_steps)):

            # print(f"------------------------------- Time step {t} -------------------------------")
               
            ref_segment = self.generate_sliding_track_reference(
                self.state,
                self.mppi_cfg.mppi_cfg.horizon,
                target_speed=desired_speed,
                dt=self.mppi_cfg.mppi_cfg.dt,
            )
            reset_nominal = False
            # action, _ = self.controller.solve(self.state, self.reference_traj, reset_nominal)
            action, info, nominal_action = self.controller.solve_total(self.state, ref_segment, reset_nominal)
            
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

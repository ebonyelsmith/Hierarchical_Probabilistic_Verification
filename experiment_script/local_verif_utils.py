import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from LCRL.env import DummyVectorEnv
from LCRL.exploration import GaussianNoise
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic
import LCRL.reach_rl_gym_envs as reach_rl_gym_envs

from gymnasium.vector import SyncVectorEnv
from copy import deepcopy
from gymnasium.vector.utils import concatenate
import matplotlib.pyplot as plt

from env_utils import NoResetSyncVectorEnv, evaluate_V_batch, find_a_batch, find_a

def make_new_env(args):
    return gym.make(args.task)

def compute_min_scenarios_alex(epsilon, delta, d):
    """
    Compute the minimum number of scenarios needed to ensure feasibility with given epsilon and delta.
    """
    # num = int(np.ceil((math.exp(1) / (epsilon*(math.exp(1)-1)))*(np.log(1/delta) + d*(d+1)/2 + d)))
    # num = torch.tensor(num, device=device)
    # num = int((2 / epsilon) * (np.log(1 / delta) + (d-1)* np.log(2)))
    num = int((2 / epsilon) * (np.log(1 / delta) + 1))
    return num

def get_nominal_trajectory2_vectorized(env, policy, initial_states, horizon, args):
    """
    Get the nominal trajectory from the environment given initial states.
    This is a vectorized version for efficiency.
    """
    num_samples = initial_states.shape[0]
    n_dim = env.observation_space.shape[0]
    state_trajs = np.zeros((num_samples, n_dim, horizon + 1))
    actions = np.zeros((num_samples, 6, horizon))

    # envs = NoResetSyncVectorEnv([make_new_env for _ in range(num_samples)])
    envs = NoResetSyncVectorEnv([lambda: make_new_env(args) for _ in range(num_samples)])
    for i, state in enumerate(initial_states):
        envs.envs[i].reset(options={'initial_state': state})

    states = np.array([env.state for env in envs.envs])
    state_trajs[:, :, 0] = states

    for t in range(horizon):
        acts = find_a_batch(states, policy)
        actions[:, :, t] = np.concatenate((acts[:, :3], np.zeros((num_samples, 3))), axis=1)  # assuming no noise in action for now
        states, _, _, _, _ = envs.step(actions[:, :, t])
        state_trajs[:, :, t + 1] = states

    return state_trajs, actions  # Shape: [num_samples, n_dim, horizon]

def get_state_trajectory2_vectorized(env, initial_states, actions, horizon, args):
    """
    Get the state trajectory from the environment given initial states and actions.
    This is a vectorized version for efficiency.
    """
    num_samples = initial_states.shape[0]
    n_dim = env.observation_space.shape[0]
    state_trajs = np.zeros((num_samples, n_dim, horizon + 1))

    # envs = NoResetSyncVectorEnv([make_new_env for _ in range(num_samples)])
    envs = NoResetSyncVectorEnv([lambda: make_new_env(args) for _ in range(num_samples)])
    for i, state in enumerate(initial_states):
        envs.envs[i].reset(options={'initial_state': state})

    states = np.array([env.state for env in envs.envs])
    state_trajs[:, :, 0] = states

    for t in range(horizon):
        states, _, _, _, _ = envs.step(actions[:, :, t])
        state_trajs[:, :, t + 1] = states

    return state_trajs  # Shape: [num_samples, n_dim, horizon]

def get_beta5(env, policy, T, epsilon_x, epsilon_d, args, gamma = 0.95, confidence=0.9, delt=1e-16):
    d = 12
    eps = 1 - confidence
    num_scenarios = compute_min_scenarios_alex(eps, delt, d)
    # print("Number of scenarios: ", num_scenarios)

    ego_vx = 0.0
    ego_vy = 0.7 # previous 0.8 ##0.2 ebonye/jingqi
    ego_z = 0.0
    ego_vz = 0.0

    # ad_x = 0.4
    # ad_vx = 0.0
    # ad_y = -2.2
    # ad_vy = 0.3
    # ad_z = 0.0
    # ad_vz = 0.0

    # Sample initial states
    # x01 = np.random.uniform(-0.9, 0.9, size=(num_scenarios, 1))
    # ego_vx1 = np.random.uniform(0, 0.1, size=(num_scenarios, 1))
    # y01 = np.random.uniform(-2.6, 0, size=(num_scenarios, 1))
    # ego_vy1 = np.random.uniform(0.6, 0.8, size=(num_scenarios, 1))
    # z01 = np.random.uniform(0, 0.1, size=(num_scenarios, 1))
    # ego_vz1 = np.random.uniform(0, 0.1, size=(num_scenarios, 1))

    # ad_x1 = np.random.uniform(0.3, 0.5, size=(num_scenarios, 1))
    # ad_vx1 = np.random.uniform(0, 0.1, size=(num_scenarios, 1))
    # ad_y1 = np.random.uniform(-2.3, -2.1, size=(num_scenarios, 1))
    # ad_vy1 = np.random.uniform(0.2, 0.4, size=(num_scenarios, 1))
    # ad_z1 = np.random.uniform(0, 0.1, size=(num_scenarios, 1))
    # ad_vz1 = np.random.uniform(0, 0.1, size=(num_scenarios, 1))
    # ego_vx1 = np.ones((num_scenarios, 1))*ego_vx
    # ego_vy1 = np.ones((num_scenarios, 1))*ego_vy
    # z01 = np.ones((num_scenarios, 1))*ego_z
    # ego_vz1 = np.ones((num_scenarios, 1))*ego_vz

    # ad_x1 = np.ones((num_scenarios, 1))*ad_x
    # ad_vx1 = np.ones((num_scenarios, 1))*ad_vx
    # ad_y1 = np.ones((num_scenarios, 1))*ad_y
    # ad_vy1 = np.ones((num_scenarios, 1))*ad_vy
    # ad_z1 = np.ones((num_scenarios, 1))*ad_z
    # ad_vz1 = np.ones((num_scenarios, 1))*ad_vz

    #######
    x01 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    ego_vx1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    y01 = np.random.uniform(-3.2, 0, size=(num_scenarios, 1))
    ego_vy1 = np.random.uniform(0.1, 1.0, size=(num_scenarios, 1))
    z01 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    ego_vz1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))

    ad_x1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    ad_vx1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    ad_y1 = np.random.uniform(-3.2, 0, size=(num_scenarios, 1))
    ad_vy1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    ad_z1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    ad_vz1 = np.random.uniform(-1.0, 1.0, size=(num_scenarios, 1))
    #########



    initial_states = np.hstack((x01, ego_vx1,
                                y01, ego_vy1,
                                z01, ego_vz1,
                                ad_x1, ad_vx1,
                                ad_y1, ad_vy1,
                                ad_z1, ad_vz1))
    
    # sample deviated initial states within epsilon_x hypercube
    # deviations = np.random.uniform(-epsilon_x, epsilon_x, size=(num_scenarios, 12))

    #sample deviated inital states within epsilon_x hypersphere
    def sample_ball(num_samples, dim, radius):
        x = np.random.normal(size=(num_samples, dim))
        dirs = x / np.linalg.norm(x, axis=1, keepdims=True)
        u = np.random.rand(num_samples, 1)
        r = radius * u**(1.0/dim)
        return dirs*r
    
    deviations = sample_ball(num_scenarios, 12, epsilon_x)
    # print(f"deviations shape: {deviations.shape}")
    # deviations_all = [np.expand_dims(deviations[:,0], axis=1), np.expand_dims(deviations[:,1], axis=1),
    #                   np.expand_dims(deviations[:,2], axis=1), np.expand_dims(deviations[:,3], axis=1),
    #                   np.expand_dims(deviations[:,4], axis=1), np.expand_dims(deviations[:,5], axis=1),
    #                   np.zeros((num_scenarios, 1)), np.zeros((num_scenarios, 1)),
    #                   np.zeros((num_scenarios, 1)), np.zeros((num_scenarios, 1)),
    #                   np.zeros((num_scenarios, 1)), np.zeros((num_scenarios, 1))]

    deviations_all = [np.expand_dims(deviations[:,0], axis=1), np.expand_dims(deviations[:,1], axis=1),
                      np.expand_dims(deviations[:,2], axis=1), np.expand_dims(deviations[:,3], axis=1),
                      np.expand_dims(deviations[:,4], axis=1), np.expand_dims(deviations[:,5], axis=1),
                      np.expand_dims(deviations[:,6], axis=1), np.expand_dims(deviations[:,7], axis=1),
                      np.expand_dims(deviations[:,8], axis=1), np.expand_dims(deviations[:,9], axis=1),
                      np.expand_dims(deviations[:,10], axis=1), np.expand_dims(deviations[:,11], axis=1)]
    
    deviations = np.hstack(deviations_all)
    # print(f"deviations shape after hstack: {deviations.shape}")
    initial_states_dev = initial_states + deviations

    # check that deviations are within epsilon_x for a few samples
    # for i in range(5):
    #     dev_norm = np.linalg.norm(initial_states_dev[i] - initial_states[i])
    #     print(f"Sample {i}: Deviation norm = {dev_norm}, Within epsilon_x: {dev_norm <= epsilon_x}")
    
    # import pdb; pdb.set_trace()

    # Get nominal trajectory
    nominal_trajs, nominal_actions = get_nominal_trajectory2_vectorized(env, policy, initial_states, T, args) 

    # Get deviated trajectory
    state_trajs = get_state_trajectory2_vectorized(env, initial_states_dev, nominal_actions, T, args)

    # Get radii
    diffs = np.linalg.norm(state_trajs - nominal_trajs, axis=1)  # Shape: [num_scenarios, horizon]

    radius = np.max(diffs, axis=0)  # Take the maximum deviation across all scenarios for each time step
    # print(f"radius: {radius}")
    # import pdb; pdb.set_trace()
    betas = radius * (gamma ** np.arange(T+1))
    
    return radius, initial_states, nominal_trajs, state_trajs, betas

def beta(T, Lf, Ld, epsilon_x, epsilon_d, certification_gamma=0.95):
    tmp = 0
    # gamma = args.gamma
    tmp = Lf**T * epsilon_x #+ (1-Lf**(T))/(1-Lf)*Ld * epsilon_d #* Lf
    # print(f"radius at time {T}: {tmp}")
    return tmp * certification_gamma**T


# plot the calibrated value function parallelized version
def calibrate_V_vectorized(env, policy, state, horizon, alphaC_list, alphaR_list, args, certification_gamma=0.95, verbose = False):
    n_dim = env.observation_space.shape[0]
    n_init_conds = state.shape[0]
    state_traj = np.zeros((n_init_conds, n_dim, horizon+1))
    state_traj[:,:,0] = state

    # envs = NoResetSyncVectorEnv([make_new_env for _ in range(n_init_conds)])
    envs = NoResetSyncVectorEnv([lambda: make_new_env(args) for _ in range(n_init_conds)])
    # for env in envs.envs:
    #     env.control_gain_2 = env.control_gain_2  # ensure opponent gain is set correctly

    for i, init_cond in enumerate(state):
        envs.envs[i].control_gain_2 = env.control_gain_2  # ensure opponent gain is set correctly
        envs.envs[i].reset(options={"initial_state": init_cond})

    value_list = np.zeros((n_init_conds, horizon))
    constraint_list = np.zeros((n_init_conds, horizon))

    for t in range(horizon):
        current_states = np.array([env.state for env in envs.envs])
        acts = find_a_batch(current_states, policy)
        # modify actions
        actions = np.concatenate((acts[:, :3], np.zeros((n_init_conds, 3))), axis=1)
        states, rew, done, _, info = envs.step(actions)
        state_traj[:, :, t+1] = states
        tmp_constraint = info["constraint"] * (certification_gamma ** t) - alphaC_list[t]
        constraint_list[:, t] = tmp_constraint
        tmp_value = np.minimum(certification_gamma ** t * rew - alphaR_list[t],
                             np.min(constraint_list[:, :t+1], axis=1))
        value_list[:, t] = tmp_value
    empirical_values = np.max(value_list, axis=1)
    time_reach_avoid = np.argmax(value_list, axis=1)
    success_flags = empirical_values > 0
    return empirical_values, time_reach_avoid, success_flags

def calibrate_V_scenario2_vectorized(env, policy, states, horizon, alphaC_list_scenario, alphaR_list_scenario, args, certification_gamma=0.95, verbose = False):
    n_dim = env.observation_space.shape[0]
    n_samples = states.shape[0]
    state_traj = np.zeros((n_samples, n_dim, horizon+1))
    state_traj[:,:,0] = states

    # envs = NoResetSyncVectorEnv([make_new_env for _ in range(n_samples)])
    envs = NoResetSyncVectorEnv([lambda: make_new_env(args) for _ in range(n_samples)])

    for i, state in enumerate(states):
        envs.envs[i].reset(options={"initial_state": state})

    value_list = np.zeros((n_samples, horizon))
    constraint_list = np.zeros((n_samples, horizon))

    for t in range(horizon):
        current_states = np.array([env.state for env in envs.envs])
        acts = find_a_batch(current_states, policy)
        # print(f"acts at time {t}: {acts}")
        # modify actions
        actions = np.concatenate((acts[:, :3], np.zeros((n_samples, 3))), axis=1)
        states, rew, done, _, info = envs.step(actions)
        # import pdb; pdb.set_trace()
        state_traj[:,:,t+1] = states
        tmp_constraint = info["constraint"] * (certification_gamma ** t) - alphaC_list_scenario[t]
        constraint_list[:, t] = tmp_constraint
        tmp_value = np.minimum(certification_gamma ** t * rew - alphaR_list_scenario[t],
                             np.min(constraint_list[:, :t+1], axis=1))
        value_list[:, t] = tmp_value
    empirical_values = np.max(value_list, axis=1)
    time_reach_avoid = np.argmax(value_list, axis=1)
    success_flags = empirical_values > 0
    # print(f"empirical_values: {empirical_values}")
    return empirical_values, time_reach_avoid, success_flags

# def calibrate_V_scenario_local_vectorized(env, policy, states, horizon, alphaC_list_scenario, alphaR_list_scenario, args, certification_gamma=0.95, verbose = False):
def calibrate_V_scenario_local_vectorized(env, policy, states, horizon, args, certification_gamma=0.95, verbose = False):
    n_dim = env.observation_space.shape[0]
    n_samples = states.shape[0]
    state_traj = np.zeros((n_samples, n_dim, horizon+1))
    state_traj[:,:,0] = states

    # envs = NoResetSyncVectorEnv([make_new_env for _ in range(n_samples)])
    envs = NoResetSyncVectorEnv([lambda: make_new_env(args) for _ in range(n_samples)])

    for i, state in enumerate(states):
        envs.envs[i].control_gain_2 = env.control_gain_2  # ensure opponent gain is set correctly
        envs.envs[i].reset(options={"initial_state": state})

    value_list = np.zeros((n_samples, horizon))
    constraint_list = np.zeros((n_samples, horizon))

    for t in range(horizon):
        current_states = np.array([env.state for env in envs.envs])
        acts = find_a_batch(current_states, policy)
        # modify actions
        actions = np.concatenate((acts[:, :3], np.zeros((n_samples, 3))), axis=1)
        states, rew, done, _, info = envs.step(actions)
        state_traj[:,:,t+1] = states
        tmp_constraint = info["constraint"] * (certification_gamma ** t) # - alphaC_list_scenario[t]
        constraint_list[:, t] = tmp_constraint
        tmp_value = np.minimum(certification_gamma ** t * rew, # - alphaR_list_scenario[t],
                             np.min(constraint_list[:, :t+1], axis=1))
        value_list[:, t] = tmp_value
    empirical_values = np.max(value_list, axis=1)
    time_reach_avoid = np.argmax(value_list, axis=1)
    success_flags = empirical_values > 0
    # import pdb; pdb.set_trace()
    return empirical_values, time_reach_avoid, success_flags

def sample_contour(vertices, num_samples):
    diffs = np.diff(vertices, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate(([0], np.cumsum(segment_lengths)))
    arc /= arc[-1]

    s = np.linspace(0, 1, num_samples)
    sampled = np.vstack([
        np.interp(s, arc, vertices[:, 0]),
        np.interp(s, arc, vertices[:, 1])
    ]).T
    return sampled

def sample_points_in_ball(center, radius, num_samples=10):
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_samples))
    x_samples = center[0] + radii * np.cos(angles)
    y_samples = center[1] + radii * np.sin(angles)
    return list(zip(x_samples, y_samples))

def max_radius_growth_vectorized_worst(current_state, seed_ii, seed_jj, X, Y, env, horizon, alphaC_list, alphaR_list,
                      V_lp_scenario_updated, policy, args, max_attept_radius = 0.5, N_samples = 20, tol=1e-2, max_iters = 10, verbose=False):
    
    # center_x = X[seed_ii, seed_jj]
    # center_y = Y[seed_ii, seed_jj]

    center_x = seed_ii
    center_y = seed_jj

    # r_min = 0.0
    # r_max = max_attept_radius
    rad = max_attept_radius
    rad_prev = 0.0
    iters = 0
    # while r_max - r_min > tol:
    points_dict = {}
    while iters < max_iters:
        iters += 1
        if verbose:
            print(f"Iteration {iters}")
        # r_candidate = (r_min + r_max) / 2.0
        points = sample_points_in_ball((center_x, center_y), rad, num_samples=N_samples)
        points_dict[iters] = points
        points_array = np.array(points)
        initial_states = np.zeros((N_samples, 12))
        ego_vx = 0.0
        ego_vy = 0.7
        ego_z = 0.0
        ego_vz = 0.0

        ad_x = 0.4
        ad_vx = 0.0
        ad_y = -2.2
        ad_vy = 0.3
        ad_z = 0.0
        ad_vz = 0.0
        initial_states[:, 0] = points_array[:, 0]
        # initial_states[:, 1] = ego_vx
        initial_states[:, 1] = current_state[1]
        initial_states[:, 2] = points_array[:, 1]
        # initial_states[:, 3] = ego_vy
        # initial_states[:, 4] = ego_z
        # initial_states[:, 5] = ego_vz
        # initial_states[:, 6] = ad_x
        # initial_states[:, 7] = ad_vx
        # initial_states[:, 8] = ad_y
        # initial_states[:, 9] = ad_vy
        # initial_states[:, 10] = ad_z
        # initial_states[:, 11] = ad_vz
        initial_states[:, 3:] = current_state[3:]

        V_vals, _, _ = calibrate_V_scenario_local_vectorized(env, policy, initial_states, horizon, args)
        # import pdb; pdb.set_trace()
        violations = V_vals <= 0
        if not np.any(violations):
            if verbose:
                print(f"Iteration {iters}: radius={rad}, all points safe.")
                return rad, points_dict
        else:
            
            
            # Shrink radius to the maximum distance of violating points
            violating_points = points_array[violations]
            if violating_points.shape[0] == 0:
                if verbose:
                    print(f"Iteration {iters}: radius={rad}, no violating points found.")
                if rad - rad_prev < tol:
                    return rad, points_dict
                rad_prev = rad

            dists = np.linalg.norm(violating_points - np.array([center_x, center_y]), axis=1)
            rad_new = dists.min()
            if verbose:
                print(f"Iteration {iters}: radius={rad}, reducing to {rad_new} based on violating points.")
            if abs(rad - rad_new) < tol:
                return rad_new, points_dict
            rad = rad_new
            
    return rad, points_dict


def grow_regions_closest_point(current_state, X, Y, env, horizon, alphaC_list, alphaR_list, policy, args,
            V_lp_scenario_updated, max_attept_radius = 0.5, N_samples = 20, tol=1e-2, target=False):
    # boundary_cells = get_boundary_cells(V_lp_scenario_updated)
    # print(f"Number of boundary: {len(boundary_cells)}")

    ### use contours1 to get boundary cells
    # contours1 = plt.contour((X), (Y), V_lp, levels=[0], linewidths=0)
    fig = plt.figure()
    if target:
        contours1 = plt.contour((X), (Y), V_lp_scenario_updated, levels=[1-1e-6, 1], linewidths=0)
        # import pdb; pdb.set_trace()
        paths = contours1.collections[0].get_paths()
        contour_points = [p.vertices for p in paths]
        boundary_cells = []
        if len(contour_points) == 0:
            return (0, 0, 0), 0, 0, {}
    else:
        contours1 = plt.contour((X), (Y), V_lp_scenario_updated, levels=[0], linewidths=0)
        paths = contours1.collections[0].get_paths()
        contour_points = [p.vertices for p in paths]
        boundary_cells = []
        if len(contour_points) == 0:
            return (0, 0, 0), 0, 0, {}
    
    plt.close(fig)

    paths = contours1.collections[0].get_paths()
    contour_points = [p.vertices for p in paths]
    boundary_cells = []
    # print(f"Number of boundary points
    # print(f"Number of boundary contours: {len(contour_points)}")
    # if len(contour_points) == 0:
    #     return (0, 0, 0), 0, 0, {}
    # print(f"V_lp_scenario_updated min: {V_lp_scenario_updated.min()}, max: {V_lp_scenario_updated.max()}")
    for contour in contour_points:
        samples = sample_contour(contour, num_samples=50)
        boundary_cells.extend(samples)
        # print(f"Number of boundary points in this contour: {samples.shape[0]}")
    boundary_cells = np.vstack(boundary_cells)

    # Find the closest boundary cell to the given point
    point = (current_state[0], current_state[2])
    velocities = np.array([current_state[1], current_state[3]])
    # closest_cell = min(boundary_cells, key=lambda cell: np.sqrt((X[cell] - point[0])**2 + (Y[cell] - point[1])**2))
    
    # # Find the closest boundary cell to the given point (too naive and doesn't consider velocity)
    # dist = np.linalg.norm(boundary_cells - np.array(point), axis=1)
    # closest_index = np.argmin(dist)
    # closest_cell = boundary_cells[closest_index]

    # # Find the closest boundary cell to the given point (consider velocity direction)
    # direxn = velocities / (np.linalg.norm(velocities) + 1e-6)
    # diff = boundary_cells - np.array(point)
    # forward_mask = np.dot(diff, direxn) > 0  # only consider points in the forward direction of velocity
    # candidates = boundary_cells[forward_mask]
    # if candidates.shape[0] == 0:
    #     candidates = boundary_cells  # if no points in forward direction, consider all points
    
    # idx = np.argmin(np.linalg.norm(candidates - np.array(point), axis=1))
    # closest_cell = candidates[idx]

    # Find closest boundary cell to the given point (consider velocity direction and cosine angle)
    direxn = velocities / (np.linalg.norm(velocities) + 1e-6)
    diff = boundary_cells - np.array(point)
    proj = np.dot(diff, direxn)

    forward_mask = proj > 0  # only consider points in the forward direction of velocity
    candidates = boundary_cells[forward_mask]

    if candidates.shape[0] == 0:
        candidates = boundary_cells  # if no points in forward direction, consider all points
    
    # cos_angle = proj / (np.linalg.norm(diff, axis=1) + 1e-6)  # cosine of angle between velocity and vector to boundary point
    # mask = cos_angle > 0.9  # only consider points that are roughly in the same direction as velocity (within ~25 degrees)

    diff_cand = candidates - np.array(point)
    proj_cand = np.dot(diff_cand, direxn)
    cos_angle_cand = proj_cand / (np.linalg.norm(diff_cand, axis=1) + 1e-6)
    mask = cos_angle_cand > 0.9

    if np.any(mask):
        candidates = candidates[mask]
    
    

    # proj[~mask] = -np.inf
    # new_candidates = candidates[mask]
    # if new_candidates.shape[0] > 0:
    #     candidates = new_candidates
    # idx = np.argmax(proj)  
    # closest_cell = candidates[idx]

    dist = np.linalg.norm(candidates - np.array(point), axis=1)
    closest_index = np.argmin(dist)
    closest_cell = candidates[closest_index]

    # import pdb; pdb.set_trace()





    seed_ii, seed_jj = closest_cell
    # print(f"Closest boundary cell to point {point} is at point ({seed_ii}), ({seed_jj})")
    # r_safe = max_radius_growth(seed_ii, seed_jj, X, Y, env, horizon, 
    #                            alphaC_list, alphaR_list,
    #                           V_lp_scenario_updated,
    #                            max_attept_radius, N_samples, tol)
    r_safe, points_dict = max_radius_growth_vectorized_worst(current_state, seed_ii, seed_jj, X, Y, env, horizon,
                               alphaC_list, alphaR_list,
                              V_lp_scenario_updated, policy, args,
                               max_attept_radius, N_samples, tol, verbose=False)
    # return (X[seed_ii, seed_jj], Y[seed_ii, seed_jj], r_safe), seed_ii, seed_jj
    return (seed_ii, seed_jj, r_safe), seed_ii, seed_jj, points_dict


def max_radius_growth_vectorized_worst_new_old(current_state, seed_ii, seed_jj, X, Y, env, horizon, alphaC_list, alphaR_list,
                      V_lp_scenario_updated, policy, args, max_attept_radius = 0.5, N_samples = 20, tol=1e-2, max_iters = 5, verbose=False):
    

    low = 0.0
    high = max_attept_radius
    best_safe_radius = 0.0
    points_dict = {}

    for it in range(max_iters):
        mid = (low + high) / 2.0
        points = sample_points_in_ball((seed_ii, seed_jj), mid, num_samples=N_samples)
        points_array = np.array(points)
        points_dict[it] = points
        initial_states = np.tile(current_state, (N_samples, 1))
        initial_states[:, 0] = points_array[:, 0]
        initial_states[:, 2] = points_array[:, 1]

        V_vals, _, _ = calibrate_V_scenario_local_vectorized(env, policy, initial_states, horizon, args)

        # if np.all(V_vals > 0):
        #     best_safe_radius = mid
        #     low = mid
        # else:
        #     high = mid
            
        # if high - low < tol:
        #     break

        # if all points are safe, just return the current radius
        if np.all(V_vals > 0):
            best_safe_radius = mid
            if verbose:
                print(f"Iteration {it+1}: radius={mid}, all points safe. Updating best_safe_radius to {best_safe_radius}")
            
            return best_safe_radius, points_dict
        else:
            high = mid
            if verbose:
                print(f"Iteration {it+1}: radius={mid}, some points violated. Reducing high to {high}")


    return best_safe_radius, points_dict


def max_radius_growth_vectorized_worst_new(current_state, seed_ii, seed_jj, X, Y, env, horizon, alphaC_list, alphaR_list,
                      V_lp_scenario_updated, policy, args, max_attept_radius = 0.5, N_samples = 20, tol=1e-2, max_iters = 5, verbose=False):
    

    low = 0.0
    high = max_attept_radius
    best_safe_radius = 0.0
    points_dict = {}

    for it in range(max_iters):
        # mid = (low + high) / 2.0
        rad = high
        points = sample_points_in_ball((seed_ii, seed_jj), rad, num_samples=N_samples)
        points_array = np.array(points)
        points_dict[it] = points
        initial_states = np.tile(current_state, (N_samples, 1))
        initial_states[:, 0] = points_array[:, 0]
        initial_states[:, 2] = points_array[:, 1]

        V_vals, _, _ = calibrate_V_scenario_local_vectorized(env, policy, initial_states, horizon, args)

        # if np.all(V_vals > 0):
        #     best_safe_radius = mid
        #     low = mid
        # else:
        #     high = mid
            
        # if high - low < tol:
        #     break

        # if all points are safe, just return the current radius
        if np.all(V_vals > 0):
            best_safe_radius = rad
            if verbose:
                print(f"Iteration {it+1}: radius={rad}, all points safe. Updating best_safe_radius to {best_safe_radius}")
            
            return best_safe_radius, points_dict
        else:
            # Shrink radius to the maximum distance of violating points
            violating_points = points_array[V_vals <= 0]
            dists = np.linalg.norm(violating_points - np.array([seed_ii, seed_jj]), axis=1)
            rad_new = dists.min()
            high = rad_new
            if verbose:
                print(f"Iteration {it+1}: radius={rad}, some points violated. Reducing high to {high}")

        
    # return zero radius if max iterations reached without finding a safe radius
    best_safe_radius = 0.0
    points_dict = {}


    return best_safe_radius, points_dict     

        




    # center_x = X[seed_ii, seed_jj]
    # center_y = Y[seed_ii, seed_jj]

    # center_x = seed_ii
    # center_y = seed_jj

    # r_min = 0.0
    # r_max = max_attept_radius
    # rad = max_attept_radius
    # rad_prev = 0.0
    # iters = 0
    # # while r_max - r_min > tol:
    # points_dict = {}
    # while iters < max_iters:
    #     iters += 1
    #     if verbose:
    #         print(f"Iteration {iters}")
    #     # r_candidate = (r_min + r_max) / 2.0
    #     points = sample_points_in_ball((center_x, center_y), rad, num_samples=N_samples)
    #     points_dict[iters] = points
    #     points_array = np.array(points)
    #     initial_states = np.zeros((N_samples, 12))
    #     ego_vx = 0.0
    #     ego_vy = 0.7
    #     ego_z = 0.0
    #     ego_vz = 0.0

    #     ad_x = 0.4
    #     ad_vx = 0.0
    #     ad_y = -2.2
    #     ad_vy = 0.3
    #     ad_z = 0.0
    #     ad_vz = 0.0
    #     initial_states[:, 0] = points_array[:, 0]
    #     # initial_states[:, 1] = ego_vx
    #     initial_states[:, 1] = current_state[1]
    #     initial_states[:, 2] = points_array[:, 1]
    #     # initial_states[:, 3] = ego_vy
    #     # initial_states[:, 4] = ego_z
    #     # initial_states[:, 5] = ego_vz
    #     # initial_states[:, 6] = ad_x
    #     # initial_states[:, 7] = ad_vx
    #     # initial_states[:, 8] = ad_y
    #     # initial_states[:, 9] = ad_vy
    #     # initial_states[:, 10] = ad_z
    #     # initial_states[:, 11] = ad_vz
    #     initial_states[:, 3:] = current_state[3:]

    #     V_vals, _, _ = calibrate_V_scenario_local_vectorized(env, policy, initial_states, horizon, args)
    #     # import pdb; pdb.set_trace()
    #     violations = V_vals <= 0
    #     if not np.any(violations):
    #         if verbose:
    #             print(f"Iteration {iters}: radius={rad}, all points safe.")
    #             return rad, points_dict
    #     else:
            
            
    #         # Shrink radius to the maximum distance of violating points
    #         violating_points = points_array[violations]
    #         if violating_points.shape[0] == 0:
    #             if verbose:
    #                 print(f"Iteration {iters}: radius={rad}, no violating points found.")
    #             if rad - rad_prev < tol:
    #                 return rad, points_dict
    #             rad_prev = rad

    #         dists = np.linalg.norm(violating_points - np.array([center_x, center_y]), axis=1)
    #         rad_new = dists.min()
    #         if verbose:
    #             print(f"Iteration {iters}: radius={rad}, reducing to {rad_new} based on violating points.")
    #         if abs(rad - rad_new) < tol:
    #             return rad_new, points_dict
    #         rad = rad_new
            
    # return rad, points_dict


def grow_regions_closest_point_new(current_state, X, Y, env, horizon, alphaC_list, alphaR_list, policy, args,
            V_lp_scenario_updated, max_attept_radius = 0.5, N_samples = 20, tol=1e-2, target=False):
    # boundary_cells = get_boundary_cells(V_lp_scenario_updated)
    # print(f"Number of boundary: {len(boundary_cells)}")

    ### use contours1 to get boundary cells
    # contours1 = plt.contour((X), (Y), V_lp, levels=[0], linewidths=0)
    # fig = plt.figure()
    from scipy.ndimage import binary_erosion
    if target:
        # contours1 = plt.contour((X), (Y), V_lp_scenario_updated, levels=[1-1e-6, 1], linewidths=0)
        # # import pdb; pdb.set_trace()
        # paths = contours1.collections[0].get_paths()
        # contour_points = [p.vertices for p in paths]
        # boundary_cells = []
        # if len(contour_points) == 0:
        #     return (0, 0, 0), 0, 0, {}
        safe_mask = V_lp_scenario_updated >= 1-1e-6
        eroded_mask = binary_erosion(safe_mask, structure=np.ones((3, 3)))
        boundary_mask = safe_mask ^ eroded_mask
        boundary_cells = np.stack((X[boundary_mask], Y[boundary_mask]), axis=-1)
        if boundary_cells.shape[0] == 0:
            return (0, 0, 0), 0, 0, {}
    else:
        # contours1 = plt.contour((X), (Y), V_lp_scenario_updated, levels=[0], linewidths=0)
        # paths = contours1.collections[0].get_paths()
        # contour_points = [p.vertices for p in paths]
        # boundary_cells = []
        # if len(contour_points) == 0:
        #     return (0, 0, 0), 0, 0, {}
        safe_mask = V_lp_scenario_updated >= 0
        eroded_mask = binary_erosion(safe_mask, structure=np.ones((3, 3)))
        boundary_mask = safe_mask ^ eroded_mask
        boundary_cells = np.stack((X[boundary_mask], Y[boundary_mask]), axis=-1)
        if boundary_cells.shape[0] == 0:
            return (0, 0, 0), 0, 0, {}
    
    # plt.close(fig)

    # paths = contours1.collections[0].get_paths()
    # contour_points = [p.vertices for p in paths]
    # boundary_cells = []
    # print(f"Number of boundary points
    # print(f"Number of boundary contours: {len(contour_points)}")
    # if len(contour_points) == 0:
    #     return (0, 0, 0), 0, 0, {}
    # print(f"V_lp_scenario_updated min: {V_lp_scenario_updated.min()}, max: {V_lp_scenario_updated.max()}")
    # for contour in contour_points:
    #     samples = sample_contour(contour, num_samples=50)
    #     boundary_cells.extend(samples)
    #     # print(f"Number of boundary points in this contour: {samples.shape[0]}")
    # boundary_cells = np.vstack(boundary_cells)

    # Find the closest boundary cell to the given point
    point = (current_state[0], current_state[2])
    velocities = np.array([current_state[1], current_state[3]])
    # closest_cell = min(boundary_cells, key=lambda cell: np.sqrt((X[cell] - point[0])**2 + (Y[cell] - point[1])**2))
    
    # # Find the closest boundary cell to the given point (too naive and doesn't consider velocity)
    # dist = np.linalg.norm(boundary_cells - np.array(point), axis=1)
    # closest_index = np.argmin(dist)
    # closest_cell = boundary_cells[closest_index]

    # # Find the closest boundary cell to the given point (consider velocity direction)
    # direxn = velocities / (np.linalg.norm(velocities) + 1e-6)
    # diff = boundary_cells - np.array(point)
    # forward_mask = np.dot(diff, direxn) > 0  # only consider points in the forward direction of velocity
    # candidates = boundary_cells[forward_mask]
    # if candidates.shape[0] == 0:
    #     candidates = boundary_cells  # if no points in forward direction, consider all points
    
    # idx = np.argmin(np.linalg.norm(candidates - np.array(point), axis=1))
    # closest_cell = candidates[idx]

    # Find closest boundary cell to the given point (consider velocity direction and cosine angle)
    direxn = velocities / (np.linalg.norm(velocities) + 1e-6)
    diff = boundary_cells - np.array(point)
    proj = np.dot(diff, direxn)

    forward_mask = proj > 0  # only consider points in the forward direction of velocity
    candidates = boundary_cells[forward_mask]

    if candidates.shape[0] == 0:
        candidates = boundary_cells  # if no points in forward direction, consider all points
    
    # cos_angle = proj / (np.linalg.norm(diff, axis=1) + 1e-6)  # cosine of angle between velocity and vector to boundary point
    # mask = cos_angle > 0.9  # only consider points that are roughly in the same direction as velocity (within ~25 degrees)

    diff_cand = candidates - np.array(point)
    proj_cand = np.dot(diff_cand, direxn)
    cos_angle_cand = proj_cand / (np.linalg.norm(diff_cand, axis=1) + 1e-6)
    mask = cos_angle_cand > 0.9

    if np.any(mask):
        candidates = candidates[mask]
    
    

    # proj[~mask] = -np.inf
    # new_candidates = candidates[mask]
    # if new_candidates.shape[0] > 0:
    #     candidates = new_candidates
    # idx = np.argmax(proj)  
    # closest_cell = candidates[idx]

    dist = np.linalg.norm(candidates - np.array(point), axis=1)
    closest_index = np.argmin(dist)
    closest_cell = candidates[closest_index]

    # import pdb; pdb.set_trace()





    seed_ii, seed_jj = closest_cell
    # print(f"Closest boundary cell to point {point} is at point ({seed_ii}), ({seed_jj})")
    # r_safe = max_radius_growth(seed_ii, seed_jj, X, Y, env, horizon, 
    #                            alphaC_list, alphaR_list,
    #                           V_lp_scenario_updated,
    #                            max_attept_radius, N_samples, tol)
    r_safe, points_dict = max_radius_growth_vectorized_worst_new(current_state, seed_ii, seed_jj, X, Y, env, horizon,
                               alphaC_list, alphaR_list,
                              V_lp_scenario_updated, policy, args,
                               max_attept_radius, N_samples, tol, verbose=False)
    # return (X[seed_ii, seed_jj], Y[seed_ii, seed_jj], r_safe), seed_ii, seed_jj
    return (seed_ii, seed_jj, r_safe), seed_ii, seed_jj, points_dict














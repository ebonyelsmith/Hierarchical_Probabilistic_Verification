"""Run the MPPI controller once and visualise the resulting trajectory."""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from testing_helpers import DroneRaceConfig, DroneRaceSimulation, DroneMPPIConfig

# python experiment_script/visualize_mppi_traj.py \
#     --value-path path/to/policy.pth \
#     --save-figure experiment_script/data/mppi_traj.png \
#     --save-gif experiment_script/data/mppi_traj.gif


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
    per_agent_state_dim = controller.per_agent_state_dim

    fig, ax = plt.subplots(figsize=(6, 6))

    # vel magnitude per agent
    max_speeds = []
    for idx in range(controller.num_agents):
        sl = slice(idx * per_agent_state_dim, (idx + 1) * per_agent_state_dim)
        vel = states[:, sl][:, [1, 3, 5]]
        speed = np.linalg.norm(vel, axis=1)
        max_speeds.append(speed.max())

    # Reference quarter-circle
    ref_xy = sim.reference[:, ::2][:, :2]
    print(f"Ref xy shape: {ref_xy.shape}")
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = []
    for idx in range(controller.num_agents):
        x, y = extract_xy(states, idx, per_agent_state_dim)
        lbl = "MPPI agent" if idx == controller.controlled_agent else f"Agent {idx}"
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
        tag = "MPPI agent" if idx == controller.controlled_agent else f"Agent {idx}"
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

        ref_line, = ax_gif.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="grey", label="Reference")
        gate_marker, = ax_gif.plot(0.0, 0.0, "s", color="black", label="Gate")

        trajectories = []
        velocities = []
        speed_components = []
        for idx in range(controller.num_agents):
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
        for idx in range(controller.num_agents):
            colour = colours[idx % len(colours)]
            lbl = "MPPI agent" if idx == controller.controlled_agent else f"Agent {idx}"
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
            artists.extend([ref_line, gate_marker])
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
    parser.add_argument(
        "--speed",
        type=float,
        default=DroneRaceConfig.target_speed,
        help="Reference tangential speed in m/s (default: 1.5).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DroneRaceConfig.radius,
        help="Quarter-circle radius in metres (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    # value path: /home/ebonyesmith/scenario_optimization/Lipschitz_Continuous_Reachability_Learning/experiment_script/pretrained_neural_networks/ra_droneracing_Game-v6/ddpg_reach_avoid_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_8_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_100/policy.pth
    ego_x = -0.76
    ego_vx = 0.0
    ego_y = -2.5
    ego_vy = 0.7
    ego_z = 0.0
    ego_vz = 0.0

    ad_x = 0.4
    ad_vx = 0.0
    ad_y = -2.2
    ad_vy = 0.3
    ad_z = 0.0
    ad_vz = 0.0
    init_cond = np.array([ego_x, ego_vx, ego_y, ego_vy, ego_z, ego_vz, ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz], dtype=np.float64)
    args = parse_args()
    race_cfg = DroneRaceConfig(
        radius=args.radius,
        target_speed=args.speed,
        duration=args.duration,
        value_path=args.value_path,
    )
    mppi_cfg = DroneMPPIConfig(controlled_agent_index=args.controlled_agent)
    # sim = DroneRaceSimulation(race_cfg, mppi_cfg, reachability_value_path=args.value_path)
    sim = DroneRaceSimulation(race_cfg, mppi_cfg, reference="learned_policy", initial_state=init_cond, reachability_value_path=args.value_path)
    
    from time import time
    start_time = time()
    data = sim.run()
    end_time = time()
    print(f"Simulation time: {end_time - start_time:.3f} seconds")
    plot_trajectories(data["states"], sim, save_path=args.save_figure, gif_path=args.save_gif)


if __name__ == "__main__":
    main()

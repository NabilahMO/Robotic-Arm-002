"""
evaluate.py
-----------
Load a trained PPO model and evaluate it on one full episode.

Usage:
    python evaluate.py --model models/ppo_arm_<timestamp>/best_model
    python evaluate.py --model models/ppo_arm_<timestamp>/best_model --vecnorm models/ppo_arm_<timestamp>/vecnorm.pkl
    python evaluate.py --model models/ppo_arm_<timestamp>/best_model --render
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

sys.path.insert(0, os.path.dirname(__file__))
from envs.arm_env import ArmTrackingEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, help="Path to saved model (no .zip)")
    p.add_argument("--vecnorm",  default=None,  help="Path to vecnorm.pkl (optional)")
    p.add_argument("--render",   action="store_true")
    p.add_argument("--episodes", type=int, default=1)
    return p.parse_args()


def make_env(render=False):
    render_mode = "human" if render else None
    return DummyVecEnv([lambda: ArmTrackingEnv(render_mode=render_mode)])


def run_episode(model, env):
    obs = env.reset()
    done = False
    ee_positions, tgt_positions, errors, rewards = [], [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        done = bool(terminated[0])

        raw_obs = obs[0]  # unwrap VecEnv dim
        ee_positions.append(raw_obs[0:3].copy())
        tgt_positions.append(raw_obs[6:9].copy())
        errors.append(info[0].get("tracking_error", 0.0))
        rewards.append(float(reward[0]))

    return (
        np.array(ee_positions),
        np.array(tgt_positions),
        np.array(errors),
        np.array(rewards),
    )


def plot_results(ee_pos, tgt_pos, errors, rewards, save_path="plots/tracking_error.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = np.arange(len(errors))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PPO Arm — Trajectory Tracking Evaluation", fontsize=14, fontweight="bold")

    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(tgt_pos[:, 0], tgt_pos[:, 1], tgt_pos[:, 2], "g--", lw=1.5, alpha=0.7, label="Target")
    ax1.plot(ee_pos[:, 0],  ee_pos[:, 1],  ee_pos[:, 2],  "r-",  lw=1.2, alpha=0.9, label="EE")
    ax1.scatter(*tgt_pos[0], c="green", s=40, zorder=5)
    ax1.scatter(*ee_pos[0],  c="red",   s=40, zorder=5)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory"); ax1.legend(fontsize=8)

    # Tracking error
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(steps, errors, color="#e67e22", lw=1.2)
    ax2.axhline(np.mean(errors), color="red", ls="--", lw=1, label=f"Mean={np.mean(errors):.4f}m")
    ax2.fill_between(steps, 0, errors, alpha=0.15, color="#e67e22")
    ax2.set_title("Tracking Error (||EE − Target||)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Distance (m)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Per-axis
    labels = ["X", "Y", "Z"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.plot(steps, tgt_pos[:, i], "--", color=col, alpha=0.6, lw=1,   label="Target")
        ax.plot(steps, ee_pos[:, i],  "-",  color=col, alpha=0.9, lw=1.2, label="EE")
        ax.set_title(f"{lbl}-axis tracking")
        ax.set_xlabel("Step"); ax.set_ylabel("Position (m)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.show()


def main():
    args = parse_args()

    # Build env
    env = make_env(render=args.render)

    # Wrap with VecNormalize if stats provided
    if args.vecnorm and os.path.exists(args.vecnorm):
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize loaded: {args.vecnorm}")
    else:
        if args.vecnorm:
            print(f"Warning: vecnorm file not found at {args.vecnorm}, running without it")

    model = PPO.load(args.model, env=env)
    print(f"Model loaded: {args.model}")
    print(f"Episode length: ~{env.envs[0].unwrapped._episode_steps} steps")
    print()

    all_errors, all_rewards = [], []
    ee_pos = tgt_pos = errors = rewards = None

    for ep in range(args.episodes):
        ee_pos, tgt_pos, errors, rewards = run_episode(model, env)
        all_errors.extend(errors)
        all_rewards.extend(rewards)

        print(f"Episode {ep + 1}:")
        print(f"  Steps          : {len(errors)}")
        print(f"  Total reward   : {rewards.sum():.3f}")
        print(f"  Mean error     : {errors.mean():.4f} m")
        print(f"  Max error      : {errors.max():.4f} m")
        print(f"  % steps < 5cm  : {(errors < 0.05).mean()*100:.1f}%")
        print(f"  % steps < 10cm : {(errors < 0.10).mean()*100:.1f}%")

    if args.episodes == 1 and ee_pos is not None:
        plot_results(ee_pos, tgt_pos, errors, rewards)

    env.close()


if __name__ == "__main__":
    main()
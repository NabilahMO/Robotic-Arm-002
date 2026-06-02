"""
train.py
--------
Train a PPO agent to track the 3D Lissajous trajectory.

Key changes from v1:
- Higher entropy coefficient (ent_coef=0.02) to prevent premature collapse
- VecNormalize wrapper for reward normalisation
- Longer rollouts (n_steps=4096) for better gradient estimates
- Saves VecNormalize stats alongside model (needed for eval)

Usage:
    python train.py                      # 500k steps
    python train.py --timesteps 1000000  # 1M steps
    python train.py --timesteps 50000    # quick sanity check
"""

import os, sys, argparse, datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(__file__))
from envs.arm_env import ArmTrackingEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",       type=int, default=500_000)
    p.add_argument("--n-envs",          type=int, default=4)
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/ppo_arm_{ts}"
    log_dir   = f"logs/ppo_arm_{ts}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    print("=" * 60)
    print(f"  PPO ARM TRAINING  v2")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Envs      : {args.n_envs}")
    print(f"  Model dir : {model_dir}")
    print("=" * 60)

    # ── Training envs with reward normalisation ────────────
    train_env = make_vec_env(ArmTrackingEnv, n_envs=args.n_envs, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # ── Eval env (no reward normalisation — raw rewards) ───
    eval_env = make_vec_env(ArmTrackingEnv, n_envs=1, seed=args.seed)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # ── PPO ───────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lambda progress: 3e-4 * (0.5 * (1 + np.cos(np.pi * (1 - progress)))),
        n_steps=4096,        # longer rollouts = better gradient estimates
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,       # higher entropy = more exploration, prevents collapse
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=log_dir,
        seed=args.seed,
        verbose=1,
    )

    print(f"\nObs dim: {train_env.observation_space.shape}")
    print(f"Act dim: {train_env.action_space.shape}")
    print(f"ent_coef: {model.ent_coef}  (higher = more exploration)\n")

    # ── Callbacks ─────────────────────────────────────────
    freq = max(args.checkpoint_freq // args.n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=freq, save_path=model_dir,
        name_prefix="ppo_arm", verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=freq,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        progress_bar=True,
    )

    # Save model + VecNormalize stats together
    final_path = os.path.join(model_dir, "ppo_arm_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnorm.pkl"))

    print(f"\nFinal model : {final_path}.zip")
    print(f"VecNorm     : {model_dir}/vecnorm.pkl")
    print(f"Evaluate    : python evaluate.py --model {final_path} --vecnorm {model_dir}/vecnorm.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
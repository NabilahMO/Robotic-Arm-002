"""
train.py
--------
Train a PPO agent to track the 3D Lissajous trajectory.

Run multiple experiments simultaneously in separate terminals:

  Terminal 1 — baseline (good config):
    python3 train.py --run-id baseline --gamma 0.99 --n-steps 4096 --batch-size 256

  Terminal 2 — test cosine LR only:
    python3 train.py --run-id cosine-lr --gamma 0.99 --n-steps 4096 --batch-size 256 --cosine-lr

  Terminal 3 — test larger rollouts:
    python3 train.py --run-id big-rollout --gamma 0.99 --n-steps 8192 --batch-size 512

  Terminal 4 — test low gamma:
    python3 train.py --run-id low-gamma --gamma 0.95 --n-steps 4096 --batch-size 256

Compare results:
  tensorboard --logdir logs/
"""

import os, sys, argparse, datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(__file__))
from envs.arm_env import ArmTrackingEnv


def parse_args():
    p = argparse.ArgumentParser()
    # Experiment identity
    p.add_argument("--run-id",    type=str,   default=None,
                   help="Label for this run (appended to model/log dir)")
    p.add_argument("--timesteps", type=int,   default=2_000_000)
    p.add_argument("--n-envs",    type=int,   default=4)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)

    # PPO hyperparameters — each independently tunable
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--cosine-lr",  action="store_true",
                   help="Anneal LR from --lr to ~0 via cosine schedule")
    p.add_argument("--n-steps",    type=int,   default=4096)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef",   type=float, default=0.02)
    p.add_argument("--smooth-coef",type=float, default=0.0,
                   help="Action smoothness penalty weight (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.run_id}_" if args.run_id else ""
    model_dir = f"models/ppo_{run_tag}{ts}"
    log_dir   = f"logs/ppo_{run_tag}{ts}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    # Learning rate — fixed or cosine annealed
    if args.cosine_lr:
        lr = lambda p: args.lr * 0.5 * (1 + np.cos(np.pi * (1 - p)))
        lr_label = f"cosine({args.lr}→0)"
    else:
        lr = args.lr
        lr_label = str(args.lr)

    print("=" * 60)
    print(f"  PPO ARM TRAINING")
    print(f"  Run ID     : {args.run_id or 'unnamed'}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  LR         : {lr_label}")
    print(f"  n_steps    : {args.n_steps}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  gamma      : {args.gamma}")
    print(f"  ent_coef   : {args.ent_coef}")
    print(f"  smooth_coef: {args.smooth_coef}")
    print(f"  Model dir  : {model_dir}")
    print("=" * 60)

    # Pass smooth_coef into env via env_kwargs
    def make_env_fn():
        env = ArmTrackingEnv(smooth_coef=args.smooth_coef)
        return env

    train_env = make_vec_env(make_env_fn, n_envs=args.n_envs, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = make_vec_env(make_env_fn, n_envs=1, seed=args.seed)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=log_dir,
        seed=args.seed,
        verbose=1,
    )

    freq = max(args.checkpoint_freq // args.n_envs, 1)
    callbacks = CallbackList([
        CheckpointCallback(save_freq=freq, save_path=model_dir,
                           name_prefix="ppo_arm", verbose=0),
        EvalCallback(eval_env, best_model_save_path=model_dir,
                     log_path=log_dir, eval_freq=freq,
                     n_eval_episodes=3, deterministic=True, verbose=1),
    ])

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(model_dir, "ppo_arm_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnorm.pkl"))

    print(f"\nSaved: {final_path}.zip")
    print(f"Eval:  python3 evaluate.py --model {model_dir}/best_model --vecnorm {model_dir}/vecnorm.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
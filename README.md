# Robotic-Arm-002 — Trajectory Tracking RL

PPO agent controlling a 3-DOF robotic arm to track a 3D Lissajous trajectory.
Built on raw MuJoCo — no gymnasium-robotics dependency.

---

## Setup

```bash
# Python 3.10+ required (3.13 recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Project structure

```
Robotic-Arm-002/
├── envs/
│   ├── arm.xml        MuJoCo model: 3-DOF arm + mocap target sphere
│   └── arm_env.py     Gymnasium env wrapper
├── trajectories/
│   └── lissajous.py   3D Lissajous trajectory generator
├── train.py           PPO training
├── evaluate.py        Evaluation + plots
└── requirements.txt
```

---

## Verify environment

```bash
python envs/arm_env.py
```

Expected output — random policy baseline, one episode:
```
Obs space  : Box(-inf, inf, (12,), float32)
Act space  : Box(-1.0, 1.0, (3,), float32)
Episode len: ~3141 steps
...
Mean error : ~0.4 m   (random policy, expect high)
```

---

## Verify trajectory

```bash
python trajectories/lissajous.py
```

---

## Train

```bash
# Sanity check (fast, ~2 min)
python train.py --timesteps 50000

# Full training (~20-40 min on M-series Mac)
python train.py --timesteps 500000

# Longer run for better tracking
python train.py --timesteps 1000000 --n-envs 8
```

Checkpoints saved to `models/ppo_arm_<timestamp>/`.
Monitor with:
```bash
tensorboard --logdir logs/
```

---

## Evaluate

```bash
# Plot tracking error (headless)
python evaluate.py --model models/ppo_arm_<timestamp>/best_model

# With MuJoCo viewer (opens a window)
python evaluate.py --model models/ppo_arm_<timestamp>/best_model --render
```

Saves `plots/tracking_error.png` with:
- 3D trajectory overlay (target vs EE)
- Per-axis tracking
- Error over time
- Reward over time

---

## Observation space (12-dim)

| Index | Meaning |
|---|---|
| 0–2 | End-effector position (xyz) |
| 3–5 | End-effector velocity (xyz) |
| 6–8 | Target position (xyz) |
| 9–11 | Target velocity (xyz) |

## Action space (3-dim, [-1, 1])

Normalised torques: `[joint0, joint1, joint2]`

## Reward

```
r = -||ee_pos - target_pos|| - 0.05 * ||ee_vel - target_vel|| - 0.001 * ||action||
```

---

## What to expect

| Training budget | Mean tracking error |
|---|---|
| 50k steps (sanity) | ~0.3–0.4 m |
| 200k steps | ~0.15–0.25 m |
| 500k steps | ~0.05–0.15 m |
| 1M steps | ~0.02–0.08 m |

A mean error under 5 cm is excellent for this task.

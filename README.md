# Robotic-Arm-002

## Overview

This project trains a reinforcement learning-based PPO agent to control a simulated 3-joint robotic arm so that its end-effector follows a continuously moving 3D target. The target traces a Lissajous curve.

The environment is built directly on MuJoCo to allow for full control over the physics model, observation space, reward function, and episode structure. Training uses Stable-Baselines3.

**Results after 2M training steps:**
- Mean tracking error: **2.3 cm**
- Steps within 5 cm of target: **95.5%**
- All three axes (X, Y, Z) tracked simultaneously

---

## Setting Up

**Requirements:** Python 3.10+ (3.13 or 3.14 recommended on macOS)

```bash
# Clone the repo
git clone https://github.com/NabilahMO/Robotic-Arm-002.git
cd Robotic-Arm-002

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install mujoco gymnasium stable-baselines3 numpy matplotlib tensorboard tqdm rich
```

**Verify the environment loads correctly:**

```bash
python3 envs/arm_env.py
python3 trajectories/lissajous.py
```

---

## Key Scripts

### `envs/arm.xml`
MuJoCo model defining the 3-DOF arm. Three hinge joints (shoulder yaw, shoulder pitch, elbow pitch) with explicit inertial properties and a mocap sphere that serves as the moving target marker.

### `envs/arm_env.py`
Custom Gymnasium environment wrapping the MuJoCo model.

| Property | Detail |
|---|---|
| Observation | 18-dim: EE pos/vel, target pos/vel, joint pos/vel |
| Action | 3-dim continuous `[-1, 1]` — normalised joint torques |
| Reward | `-dist - 0.1·vel_err - 0.005·ctrl - 0.5·limit_penalty + 0.5·(if dist < 5cm)` |
| Episode length | One Lissajous period (~897 env steps, ~9 sim seconds) |

### `trajectories/lissajous.py`
Generates the 3D Lissajous target trajectory. Parameterised by amplitude, frequency, and phase per axis. Returns position and velocity at any timestep.

```
x(t) = cx + Ax·sin(ax·t + dx)
y(t) = cy + Ay·sin(ay·t + dy)
z(t) = cz + Az·sin(az·t + dz)
```

### `train.py`
PPO training script using Stable-Baselines3.

Key hyperparameters: `lr=3e-4`, `ent_coef=0.02`, `n_steps=4096`, `batch_size=256`, network `[256, 256]` for both actor and critic.

Saves checkpoints to `models/ppo_arm_<timestamp>/` and logs to `logs/` for TensorBoard.

### `evaluate.py`
Loads a trained model and runs one evaluation episode. Prints tracking statistics and saves a plot.

```bash
python3 evaluate.py --model models/ppo_arm_<timestamp>/best_model \
                    --vecnorm models/ppo_arm_<timestamp>/vecnorm.pkl
```

---

## Expected Outputs

| Training budget | Mean tracking error | Notes |
|---|---|---|
| 50k steps | ~0.35–0.50 m | Random-policy baseline, arm barely moving |
| 200k steps | ~0.25–0.35 m | Arm starts following trajectory shape |
| 500k steps | ~0.15–0.25 m | X and Y tracking, Z still lagging |
| 2M steps | **~0.02–0.05 m** | All three axes tracking, >95% within 5cm |

After a full 2M step run, `evaluate.py` produces `plots/tracking_error.png` containing:
- 3D overlay of EE path vs target path
- Per-axis tracking (X, Y, Z)
- Tracking error over the episode
- Reward per step

---

## References

- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Todorov et al. (2012) — [MuJoCo: A physics engine for model-based control](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
- Raffin et al. (2021) — [Stable-Baselines3: Reliable Reinforcement Learning Implementations](https://jmlr.org/papers/v22/20-1364.html)
- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [Gymnasium Documentation](https://gymnasium.farama.org)
- Claude Code was used for code execution which was subsequently modified 

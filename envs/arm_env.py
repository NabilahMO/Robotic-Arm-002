"""
envs/arm_env.py
---------------
Custom Gymnasium environment: 3-DOF arm tracking a 3D Lissajous trajectory.

Observation (12-dim, normalised):
    [0:3]  end-effector position (xyz)
    [3:6]  end-effector velocity (xyz)
    [6:9]  target position       (xyz)
    [9:12] target velocity       (xyz)

Action (3-dim, continuous [-1, 1]):
    Normalised torques applied to joint0, joint1, joint2.

Reward (per step):
    r = -dist
        - LAMBDA_VEL  * ||ee_vel - target_vel||
        - LAMBDA_CTRL * ||action||
        + BONUS_CLOSE  (if dist < 0.05m)

Episode:
    Fixed length of one Lissajous period.
    Always resets to same initial qpos so agent learns one trajectory.
"""

import os
import sys
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajectories.lissajous import LissajousTrajectory

# ── Constants ─────────────────────────────────────────────
XML_PATH     = os.path.join(os.path.dirname(__file__), "arm.xml")
LAMBDA_VEL   = 0.1       # velocity matching weight
LAMBDA_CTRL  = 0.005     # action magnitude penalty
BONUS_CLOSE  = 0.5       # bonus reward when within 5cm
FRAME_SKIP   = 5         # physics steps per env step
OBS_DIM      = 12
ACT_DIM      = 3

# Initial joint config: arm pointing up and slightly forward
INIT_QPOS = np.array([0.0, 0.3, -0.6])
INIT_QVEL = np.zeros(3)

# Obs normalisation constants (approximate scale of each component)
OBS_SCALE = np.array([
    1.0, 1.0, 1.0,   # ee pos  (metres, ~0–1 range)
    2.0, 2.0, 2.0,   # ee vel  (m/s)
    1.0, 1.0, 1.0,   # tgt pos (metres)
    0.5, 0.5, 0.5,   # tgt vel (m/s)
], dtype=np.float32)


class ArmTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self._ee_site_id = self.model.site("ee_site").id
        self._mocap_id   = 0

        self.traj = LissajousTrajectory(dt=self.model.opt.timestep)
        self._episode_steps = self.traj.steps_per_period() // FRAME_SKIP
        self._step_count    = 0

        obs_high = np.ones(OBS_DIM, dtype=np.float32) * 5.0  # normalised, ±5 is plenty
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

        self.render_mode = render_mode
        self._renderer   = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    # ── Gymnasium API ─────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = INIT_QPOS
        self.data.qvel[:] = INIT_QVEL
        self._step_count  = 0
        self._update_target(0)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        sim_time = self._step_count * FRAME_SKIP * self.model.opt.timestep
        self._update_target(sim_time)
        mujoco.mj_forward(self.model, self.data)

        obs    = self._get_obs()
        reward = self._compute_reward(action)
        info   = {"tracking_error": self._tracking_error()}

        return obs, reward, False, self._step_count >= self._episode_steps, info

    def render(self):
        if self.render_mode == "human" and self._renderer:
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    # ── Internal ──────────────────────────────────────────
    def _update_target(self, t):
        pos, _ = self.traj.get(t)
        self.data.mocap_pos[self._mocap_id] = pos

    def _get_ee_pos(self):
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_ee_vel(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        return jacp @ self.data.qvel

    def _get_obs(self):
        sim_time = self._step_count * FRAME_SKIP * self.model.opt.timestep
        tgt_pos, tgt_vel = self.traj.get(sim_time)
        raw = np.concatenate([
            self._get_ee_pos(), self._get_ee_vel(),
            tgt_pos, tgt_vel
        ]).astype(np.float32)
        return raw / OBS_SCALE   # normalise

    def _tracking_error(self):
        sim_time = self._step_count * FRAME_SKIP * self.model.opt.timestep
        tgt_pos, _ = self.traj.get(sim_time)
        return float(np.linalg.norm(self._get_ee_pos() - tgt_pos))

    def _compute_reward(self, action):
        sim_time = self._step_count * FRAME_SKIP * self.model.opt.timestep
        tgt_pos, tgt_vel = self.traj.get(sim_time)

        ee_pos = self._get_ee_pos()
        ee_vel = self._get_ee_vel()

        dist    = np.linalg.norm(ee_pos - tgt_pos)
        vel_err = np.linalg.norm(ee_vel - tgt_vel)
        ctrl    = np.linalg.norm(action)

        reward  = -dist
        reward -= LAMBDA_VEL  * vel_err
        reward -= LAMBDA_CTRL * ctrl
        if dist < 0.05:
            reward += BONUS_CLOSE   # encourage tight tracking

        return float(reward)


if __name__ == "__main__":
    env = ArmTrackingEnv()
    obs, _ = env.reset()
    print(f"Obs space  : {env.observation_space}")
    print(f"Act space  : {env.action_space}")
    print(f"Episode len: {env._episode_steps} steps")
    print(f"Init obs   : {obs}")

    total_r, errors = 0.0, []
    for _ in range(env._episode_steps):
        obs, r, _, trunc, info = env.step(env.action_space.sample())
        total_r += r
        errors.append(info["tracking_error"])
        if trunc: break

    print(f"\nRandom policy: reward={total_r:.1f}  mean_err={np.mean(errors):.4f}m")
    env.close()
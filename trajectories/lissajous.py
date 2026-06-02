"""
trajectories/lissajous.py
--------------------------
Generates a 3D Lissajous trajectory for the arm to track.

A Lissajous curve in 3D:
    x(t) = cx + Ax * sin(ax * t + dx)
    y(t) = cy + Ay * sin(ay * t + dy)
    z(t) = cz + Az * sin(az * t + dz)

With ax != ay != az the path is a non-repeating 3D figure that
fills a box — visually interesting and a good tracking challenge.

Usage:
    traj = LissajousTrajectory()
    pos, vel = traj.get(t)      # t in seconds
    pos, vel = traj.get_step(n) # n in timesteps
"""

import numpy as np


class LissajousTrajectory:
    """
    3D Lissajous trajectory centred in the arm's workspace.

    Parameters
    ----------
    cx, cy, cz   : centre of the trajectory (metres)
    Ax, Ay, Az   : amplitudes per axis (metres)
    ax, ay, az   : angular frequencies per axis (rad/s)
    dx, dy, dz   : phase offsets per axis (rad)
    dt           : simulation timestep (must match arm.xml)
    """

    def __init__(
        self,
        cx=0.0,  cy=0.0,  cz=0.72,   # centre — in arm workspace
        Ax=0.30, Ay=0.30, Az=0.18,    # amplitudes: ±30 cm xy, ±18 cm z
        ax=1.0,  ay=1.3,  az=0.7,     # incommensurate freqs → non-repeating path
        dx=0.0,  dy=np.pi/4, dz=np.pi/2,
        dt=0.002,
    ):
        self.centre = np.array([cx, cy, cz])
        self.amp    = np.array([Ax, Ay, Az])
        self.freq   = np.array([ax, ay, az])
        self.phase  = np.array([dx, dy, dz])
        self.dt     = dt

    # ── Core ──────────────────────────────────────────────
    def get(self, t: float):
        """
        Returns (position, velocity) at time t (seconds).
        Both are np.ndarray shape (3,).
        """
        theta = self.freq * t + self.phase
        pos   = self.centre + self.amp * np.sin(theta)
        vel   = self.amp * self.freq * np.cos(theta)   # d/dt of pos
        return pos, vel

    def get_step(self, step: int):
        """Convenience wrapper: step index → (pos, vel)."""
        return self.get(step * self.dt)

    # ── Utility ───────────────────────────────────────────
    def period(self) -> float:
        """
        Approximate period of the full 3D pattern (seconds).
        Uses the LCM of the three frequencies (approximate for
        irrational ratios — gives one 'visual' cycle).
        """
        return 2 * np.pi / min(self.freq)

    def steps_per_period(self) -> int:
        return int(self.period() / self.dt)

    def preview(self, n_steps: int = None) -> np.ndarray:
        """
        Returns array of shape (n_steps, 3) — trajectory positions.
        Useful for plotting or sanity-checking the workspace.
        """
        if n_steps is None:
            n_steps = self.steps_per_period()
        return np.array([self.get_step(i)[0] for i in range(n_steps)])


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    traj = LissajousTrajectory()

    print(f"Period      : {traj.period():.2f} s")
    print(f"Steps/period: {traj.steps_per_period()}")
    print()

    # Sample a few points
    for t in [0, 1, 2, 3]:
        pos, vel = traj.get(t)
        print(f"t={t}s  pos={pos.round(3)}  vel={vel.round(3)}")

    # Check bounding box
    pts = traj.preview(n_steps=10000)
    print(f"\nBounding box:")
    print(f"  x: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

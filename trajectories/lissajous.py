"""
trajectories/lissajous.py
--------------------------
3D Lissajous trajectory sized to fit comfortably inside the arm's workspace.

Workspace (from arm.xml joint sweep):
  x: [-0.52, 0.52]
  y: [-0.52, 0.52]
  z: [-0.20,  0.62]

Trajectory kept well inside these bounds with margin for dynamics.
"""

import numpy as np


class LissajousTrajectory:
    """
    x(t) = cx + Ax * sin(ax * t + dx)
    y(t) = cy + Ay * sin(ay * t + dy)
    z(t) = cz + Az * sin(az * t + dz)

    Frequencies are incommensurate so the path never exactly repeats.
    """

    def __init__(
        self,
        cx=0.0,  cy=0.0,  cz=0.50,   # centre — reachable by all joint configs
        Ax=0.20, Ay=0.20, Az=0.10,    # reduced amplitudes — fits inside workspace
        ax=1.0,  ay=1.3,  az=0.7,     # incommensurate frequencies
        dx=0.0,  dy=np.pi/4, dz=np.pi/2,
        dt=0.002,
    ):
        self.centre = np.array([cx, cy, cz])
        self.amp    = np.array([Ax, Ay, Az])
        self.freq   = np.array([ax, ay, az])
        self.phase  = np.array([dx, dy, dz])
        self.dt     = dt

    def get(self, t: float):
        theta = self.freq * t + self.phase
        pos   = self.centre + self.amp * np.sin(theta)
        vel   = self.amp * self.freq * np.cos(theta)
        return pos, vel

    def get_step(self, step: int):
        return self.get(step * self.dt)

    def period(self) -> float:
        return 2 * np.pi / min(self.freq)

    def steps_per_period(self) -> int:
        return int(self.period() / self.dt)

    def preview(self, n_steps=None) -> np.ndarray:
        if n_steps is None:
            n_steps = self.steps_per_period()
        return np.array([self.get_step(i)[0] for i in range(n_steps)])


if __name__ == "__main__":
    traj = LissajousTrajectory()
    print(f"Period      : {traj.period():.2f} s")
    print(f"Steps/period: {traj.steps_per_period()}")
    pts = traj.preview(10000)
    print(f"\nBounding box:")
    print(f"  x: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
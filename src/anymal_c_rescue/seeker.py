# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the velocity command from the thermal camera instead of from a random number.

Why this exists: the rescue task inherited ``AnymalCRoughEnvCfg``'s command manager unchanged,
so ``base_velocity`` was a ``UniformVelocityCommand`` drawing a fresh random direction and
heading every 10 s. The robot was never searching -- it was obeying a random walk while the
victim rewards nudged its gait from the sidelines. The sensors reached the observation vector
but nothing let the policy act on them to *choose a destination*. See the closing section of
``artifacts/benchmarks/phase3_rescue.md``.

This closes that loop the cheap way, with no training at all:

    thermal image -> hottest pixels -> bearing -> velocity command -> existing walker

It works because the trained policy is already a good velocity tracker (``Metrics/success_rate
= 0.8433`` on command following). Point the command at a casualty and it walks there. The
locomotion checkpoint is reused untouched.

**This is a hand-written controller, not learned search, and the write-up must say so.** It is
also what fielded systems commonly do -- the ANYmal entries in the DARPA SubT Challenge ran
classical planning over a locomotion layer. Its value here is twofold: it makes the sensor
suite load-bearing for the first time, and it gives a learned navigator (the hierarchical
option) a baseline to be measured against, without which that number means little.

Known limits, all inherent to being stateless:

* **No memory.** A casualty that goes behind rubble is forgotten instantly and the robot
  reverts to scanning. A recurrent navigator would not.
* **No planning.** Obstacle handling is a reactive deflection, not a route around.
* **Proxy objective.** It optimises "point at the hot pixels", not "find casualties".
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

from . import thermal as thermal_utils

# Midway between ambient (12.0 C) and a casualty (36.5 C). The gap is 24.5 C, so the exact
# threshold is not delicate -- anything clear of sensor noise separates them. Stated as a
# temperature rather than a normalized value so it stays readable against
# CLASS_TEMPERATURE_C.
DETECT_TEMPERATURE_C: float = 24.0


class VictimSeekingCommand(UniformVelocityCommand):
    """Velocity command that points at the warmest thing the thermal camera can see.

    Subclasses ``UniformVelocityCommand`` rather than reimplementing ``CommandTerm``: the
    parent already owns the buffer shapes, the metrics, the debug arrow visualisers and the
    standing-envs handling, and all of that should keep working. Only the question of *what
    the command should be* is overridden.

    The parent's random resampling still runs and is then overwritten every step. That is
    deliberate -- it keeps ``_resample_command`` supplying a well-formed buffer to fall back
    on, rather than special-casing the first frame before any image exists.
    """

    cfg: VictimSeekingCommandCfg

    def __init__(self, cfg: VictimSeekingCommandCfg, env):
        super().__init__(cfg, env)
        self._camera = env.scene[cfg.camera_name]
        # Horizontal half-FOV from the lens, not hardcoded: focal_length and aperture live in
        # the camera's spawn cfg, and a copy of "30 degrees" here would rot the moment someone
        # changed the lens.
        spawn = self._camera.cfg.spawn
        aperture = getattr(spawn, "horizontal_aperture", 20.955)
        self._half_fov_rad = math.atan(aperture / (2.0 * spawn.focal_length))

    def __str__(self) -> str:
        return (
            f"VictimSeekingCommand: thermal-driven, camera={self.cfg.camera_name}, "
            f"half-FOV={math.degrees(self._half_fov_rad):.1f} deg, "
            f"detect>{DETECT_TEMPERATURE_C} C"
        )

    def _update_command(self):
        """Overwrite the random command with one derived from the thermal image."""
        super()._update_command()  # keeps heading-mode and standing-env bookkeeping intact

        seg = self._camera.data.output.get("semantic_segmentation")
        if seg is None:
            return  # no image yet on the very first tick; the random command stands
        temp = thermal_utils.segmentation_to_temperature(seg.torch)  # (N, H, W)

        bearing, detected = bearing_to_hotspot(temp, self._half_fov_rad)

        # Turn toward the casualty, proportional to how far off-axis it is; walk forward.
        # When nothing is visible, stop translating and rotate on the spot -- that spin IS the
        # search behaviour, and it is exactly what the random-command version could never do.
        turn = torch.where(
            detected,
            (self.cfg.turn_gain * bearing).clamp(-self.cfg.max_ang_vel, self.cfg.max_ang_vel),
            torch.full_like(bearing, self.cfg.scan_ang_vel),
        )
        forward = torch.where(detected, torch.full_like(bearing, self.cfg.approach_vel), torch.zeros_like(bearing))

        self.vel_command_b[:, 0] = forward
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = turn


def bearing_to_hotspot(
    temperature_c: torch.Tensor,
    half_fov_rad: float,
    threshold_c: float = DETECT_TEMPERATURE_C,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal bearing to the warm pixels in each image.

    Split out from the command term so it is testable without Isaac Sim, which is the only
    reason the maths below has a self-check at all.

    Args:
        temperature_c: Temperature images, shape ``(N, H, W)`` in deg C.
        half_fov_rad: Half of the camera's horizontal field of view [rad].
        threshold_c: Pixels at or above this count as a casualty.

    Returns:
        ``(bearing, detected)``. ``bearing`` is the angle [rad] from the camera's forward axis
        to the hot centroid, **positive to the left** to match the robot's +z yaw convention,
        zero where nothing was detected. ``detected`` is a bool mask, shape ``(N,)``.
    """
    hot = temperature_c >= threshold_c  # (N, H, W)
    weight = hot.float()
    per_col = weight.sum(dim=1)  # (N, W) -- collapse rows; only the column matters for yaw
    total = per_col.sum(dim=1)  # (N,)
    detected = total > 0

    width = temperature_c.shape[-1]
    # Pixel centres in [0, 1]: the +0.5 matters at 16 px wide, where treating a pixel as its
    # left edge biases every bearing by half a pixel (~1.9 deg) toward the left.
    centres = (torch.arange(width, device=temperature_c.device, dtype=torch.float32) + 0.5) / width
    centroid = (per_col * centres).sum(dim=1) / total.clamp(min=1.0)  # (N,) in [0, 1]

    # 0.5 is dead ahead. Negate so a blob on the LEFT half (centroid < 0.5) gives a positive
    # bearing, because image +x runs right while robot +yaw turns left. Getting this backwards
    # produces a robot that flees every casualty it sees, which is the kind of bug that looks
    # like a broken policy rather than a sign error.
    bearing = -(centroid - 0.5) * 2.0 * half_fov_rad
    return torch.where(detected, bearing, torch.zeros_like(bearing)), detected


@configclass
class VictimSeekingCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`VictimSeekingCommand`."""

    class_type: type = VictimSeekingCommand

    camera_name: str = MISSING
    """Scene entity name of the thermal camera to steer by."""

    approach_vel: float = 0.8
    """Forward speed while a casualty is in view [m/s].

    Below the 1.0 m/s ceiling the policy trained against, so it sits inside the command
    distribution the walker knows rather than at its edge.
    """

    turn_gain: float = 1.5
    """Proportional gain from bearing error [rad] to yaw rate [rad/s]."""

    max_ang_vel: float = 1.0
    """Yaw-rate clamp [rad/s]. Matches the ``ang_vel_z`` training range of (-1, 1), so the
    command never leaves the distribution the walker was trained to track."""

    scan_ang_vel: float = 0.6
    """Yaw rate when nothing is detected [rad/s] -- the search sweep.

    One full rotation takes ~10 s against a ~20 s episode, so a robot that starts facing away
    still sweeps the full circle with time left to walk. Faster would scan more of the circle
    but risks yawing past a 1-2 pixel casualty between camera updates, since the camera ticks
    at a quarter of the policy rate."""


def _self_check() -> None:
    """Assert the bearing maths, because a sign error here inverts the whole behaviour."""
    half_fov = math.radians(30.2)  # matches focal_length=18.0, aperture=20.955

    def img(cols, width=16, height=16, hot=36.5, cold=12.0):
        t = torch.full((1, height, width), cold)
        for c in cols:
            t[0, :, c] = hot
        return t

    # Centred blob -> straight ahead.
    b, d = bearing_to_hotspot(img([7, 8]), half_fov)
    assert bool(d[0]) and abs(float(b[0])) < 1e-6, float(b[0])

    # Blob on the LEFT of the image -> positive (turn left).
    b, d = bearing_to_hotspot(img([0, 1]), half_fov)
    assert bool(d[0]) and float(b[0]) > 0, float(b[0])

    # Blob on the RIGHT -> negative (turn right). This is the sign that, if flipped, makes the
    # robot run away from every casualty.
    b, d = bearing_to_hotspot(img([14, 15]), half_fov)
    assert bool(d[0]) and float(b[0]) < 0, float(b[0])

    # Extremes stay inside the field of view.
    b, _ = bearing_to_hotspot(img([0]), half_fov)
    assert abs(float(b[0])) <= half_fov + 1e-6, float(b[0])

    # Nothing warm -> not detected, and bearing pinned at zero rather than NaN from 0/0.
    b, d = bearing_to_hotspot(img([]), half_fov)
    assert not bool(d[0]) and float(b[0]) == 0.0

    # The robot's own chassis at 30 C would trip a naive "warmer than ambient" test; 24 C is
    # below it, so confirm the threshold is what excludes it -- it must NOT be detected here.
    warm_robot = torch.full((1, 16, 16), 12.0)
    warm_robot[0, :, 3] = 30.0
    _, d = bearing_to_hotspot(warm_robot, half_fov, threshold_c=32.0)
    assert not bool(d[0]), "threshold above robot temperature should reject the chassis"

    # Batch: independent per image, and a mixed batch keeps its per-row detection flags.
    batch = torch.cat([img([1]), img([14]), img([])], dim=0)
    b, d = bearing_to_hotspot(batch, half_fov)
    assert d.tolist() == [True, True, False]
    assert float(b[0]) > 0 and float(b[1]) < 0 and float(b[2]) == 0.0

    print("seeker.py self-check passed")


if __name__ == "__main__":
    _self_check()

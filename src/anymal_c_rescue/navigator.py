# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learned navigator: a residual correction on top of the scripted seeker.

The stack this completes::

    thermal ---> scripted seeker ---> base command --+
                                                     +--> sum --> SHIELD --> walker
    thermal + LiDAR ---> navigator (MLP) --> delta --+                       (frozen)

Each layer has one job and can be tested alone. The seeker guarantees a floor, the network
supplies the learning, the shield guarantees safety, and ``model_4197.pt`` does locomotion.

**Residual, not raw command, and that is the whole point.** With the output initialised near
zero the policy *starts* behaving exactly like :mod:`.seeker`, so training can only move away
from a known-decent controller rather than from noise. This matters here because the training
budget affords roughly one run: if RL learns nothing useful, the result is still
scripted-level rather than nothing. The technique is from Johannink et al., *Residual
Reinforcement Learning for Robot Control* (ICRA 2019), which exists for exactly this case.

It also collapses the baseline comparison into a single run. The policy at iteration 0 **is**
the scripted controller, so evaluating the same checkpoint before and after training gives
both rows without paying for a separate instance.

Why a residual rather than letting the network learn the command outright: the useful part of
this task -- point at the warm pixels -- is already known in closed form, and spending the
first thousand iterations rediscovering it wastes the budget on the part that was never the
research question.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.utils.configclass import configclass

from .seeker import VictimSeekingCommand, VictimSeekingCommandCfg

# Ray index bookkeeping for the LiDAR shield. The pattern is 8 channels x 36 azimuth steps at
# 10 deg, flattened channel-major by ``lidar_pattern``.
_LIDAR_CHANNELS: int = 8
_LIDAR_AZIMUTH_STEPS: int = 36


def forward_ray_mask(num_rays: int, half_angle_deg: float = 25.0) -> torch.Tensor:
    """Boolean mask selecting rays pointing roughly forwards.

    Kept as a free function so the index arithmetic is testable without a simulator -- an
    off-by-one in the azimuth wrap would silently shield the wrong direction, which looks
    like a policy that refuses to walk rather than like a masking bug.

    Args:
        num_rays: Total rays in the flattened scan.
        half_angle_deg: Half-width of the forward cone to guard [deg].

    Returns:
        Bool tensor, shape ``(num_rays,)``.
    """
    steps = _LIDAR_AZIMUTH_STEPS
    res_deg = 360.0 / steps
    idx = torch.arange(num_rays)
    azimuth_idx = idx % steps
    # horizontal_fov_range is (-180, 180), so index 0 is -180 deg and the forward direction
    # (0 deg) sits at the middle of the sweep, not at index 0.
    angle = -180.0 + azimuth_idx.float() * res_deg
    return angle.abs() <= half_angle_deg


class ResidualNavigatorCommand(VictimSeekingCommand):
    """Scripted seeker plus a learned delta, with a hard LiDAR shield on top.

    The delta is written in by the training wrapper each time the navigator acts; this class
    only owns combining the three pieces. Keeping the combination here rather than in the
    wrapper means the eval and render harnesses get identical behaviour without duplicating
    it, which is how the scripted and learned versions stay comparable.
    """

    cfg: ResidualNavigatorCommandCfg

    def __init__(self, cfg: ResidualNavigatorCommandCfg, env):
        super().__init__(cfg, env)
        self._delta = torch.zeros(env.num_envs, 3, device=env.device)
        self._lidar = env.scene[cfg.lidar_name] if cfg.lidar_name else None
        self._fwd_mask: torch.Tensor | None = None
        self._shield_fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def set_delta(self, delta: torch.Tensor) -> None:
        """Install the navigator's correction. Zero delta reproduces the scripted seeker."""
        self._delta[:] = delta

    @property
    def shield_fired(self) -> torch.Tensor:
        """Per-env flag: did the LiDAR shield veto forward motion on the last update?

        Exposed so training can log how often the shield engages. A rate that stays high late
        in training means the navigator never learned to avoid obstacles and is simply being
        rescued by the shield every time -- worth knowing before claiming it learned anything.
        """
        return self._shield_fired

    def _update_command(self):
        super()._update_command()  # scripted seeker writes the base command

        if self.cfg.use_residual:
            self.vel_command_b[:, 0] += self._delta[:, 0]
            self.vel_command_b[:, 1] += self._delta[:, 1]
            self.vel_command_b[:, 2] += self._delta[:, 2]
            # Clamp to the ranges the walker was trained against. Outside them the command is
            # not merely aggressive, it is out of distribution for the low-level policy, and
            # the failure looks like bad navigation rather than an unfollowable order.
            self.vel_command_b[:, 0].clamp_(-self.cfg.max_lin_vel, self.cfg.max_lin_vel)
            self.vel_command_b[:, 1].clamp_(-self.cfg.max_lin_vel, self.cfg.max_lin_vel)
            self.vel_command_b[:, 2].clamp_(-self.cfg.max_ang_vel, self.cfg.max_ang_vel)

        self._apply_shield()

    def _apply_shield(self):
        """Veto forward motion when the LiDAR sees something close ahead.

        Outside the network and after the sum, so no amount of training can remove it. The
        turn component is deliberately left alone -- zeroing that too would freeze the robot
        against an obstacle instead of letting it turn away.
        """
        self._shield_fired[:] = False
        if self._lidar is None or not self.cfg.shield_distance > 0.0:
            return
        # (N, R, 3) hit points in world frame; distance is what matters, not direction.
        hits = self._lidar.data.ray_hits_w.torch
        origin = self._lidar.data.pos_w.torch.unsqueeze(1)
        dist = torch.norm(hits - origin, dim=-1)  # (N, R)
        # Misses come back as inf: they are clear space, not obstacles at zero range.
        dist = torch.nan_to_num(dist, nan=float("inf"), posinf=float("inf"))

        if self._fwd_mask is None or self._fwd_mask.numel() != dist.shape[1]:
            self._fwd_mask = forward_ray_mask(dist.shape[1], self.cfg.shield_half_angle_deg).to(dist.device)

        blocked = (dist[:, self._fwd_mask] < self.cfg.shield_distance).any(dim=1)
        # Only veto *forward* motion. Reversing away from an obstacle stays allowed.
        veto = blocked & (self.vel_command_b[:, 0] > 0.0)
        self.vel_command_b[veto, 0] = 0.0
        self._shield_fired[:] = veto


@configclass
class ResidualNavigatorCommandCfg(VictimSeekingCommandCfg):
    """Config for :class:`ResidualNavigatorCommand`."""

    class_type: type = ResidualNavigatorCommand

    lidar_name: str = MISSING
    """Scene entity name of the LiDAR used by the safety shield."""

    use_residual: bool = True
    """Set False to evaluate the pure scripted seeker -- this is how the baseline row of the
    comparison is produced from the same code path, so the two differ in exactly one thing."""

    max_lin_vel: float = 1.0
    max_ang_vel: float = 1.0
    """Command clamps, matching the (-1, 1) ranges the walker trained on."""

    shield_distance: float = 1.0
    """Veto forward motion when a forward ray is closer than this [m].

    1.0 m against a 0.8 m block and a ~1.0 m robot: far enough to stop before contact at
    0.8 m/s given the navigator's 5-10 Hz decision rate, close enough that the robot can still
    walk through the gaps a 4-block field leaves. Set 0 to disable the shield entirely.
    """

    shield_half_angle_deg: float = 25.0
    """Half-width of the guarded forward cone [deg]. Wider than the body so a corner clip
    counts, narrower than the thermal FOV so a casualty ahead does not permanently veto
    approach -- casualties are 0.30 m tall and the fan is tilted down, so they do register."""


def _self_check() -> None:
    """Assert the shield's ray indexing, where an off-by-one is invisible until it matters."""
    n = _LIDAR_CHANNELS * _LIDAR_AZIMUTH_STEPS  # 288, matching LidarPatternCfg
    mask = forward_ray_mask(n, half_angle_deg=25.0)

    assert mask.shape == (n,), mask.shape
    # 25 deg either side at 10 deg resolution -> indices at -20, -10, 0, 10, 20 -> 5 per channel.
    per_channel = int(mask[:_LIDAR_AZIMUTH_STEPS].sum())
    assert per_channel == 5, per_channel
    assert int(mask.sum()) == 5 * _LIDAR_CHANNELS, int(mask.sum())

    # Forward (0 deg) is index 18, since index 0 is -180 deg at 10 deg steps.
    assert bool(mask[18]), "forward ray must be inside the shield cone"
    # Directly behind must NOT be shielded, or the robot refuses to walk away from anything.
    assert not bool(mask[0]), "rear ray (-180 deg) must be outside the cone"
    assert not bool(mask[9]), "-90 deg (left) must be outside the cone"
    assert not bool(mask[27]), "+90 deg (right) must be outside the cone"

    # The mask must repeat identically for every vertical channel.
    first = mask[:_LIDAR_AZIMUTH_STEPS]
    for ch in range(1, _LIDAR_CHANNELS):
        seg = mask[ch * _LIDAR_AZIMUTH_STEPS : (ch + 1) * _LIDAR_AZIMUTH_STEPS]
        assert bool((seg == first).all()), f"channel {ch} mask differs"

    # A narrow cone must still include straight ahead and nothing else.
    narrow = forward_ray_mask(n, half_angle_deg=5.0)
    assert int(narrow[:_LIDAR_AZIMUTH_STEPS].sum()) == 1 and bool(narrow[18])

    print("navigator.py self-check passed")


if __name__ == "__main__":
    _self_check()

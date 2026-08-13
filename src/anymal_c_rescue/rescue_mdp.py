# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rescue-specific MDP terms (observations + rewards).

These are additive: the env cfg still pulls all locomotion terms from
``isaaclab_tasks.manager_based.locomotion.velocity.mdp``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCaster

from . import thermal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


##
# Helpers
##

def _victim_positions_w(env: ManagerBasedEnv, victim_names: tuple[str, ...]) -> torch.Tensor:
    """World positions of the victim props, read straight from the physics state.

    Args:
        env: The environment.
        victim_names: Scene entity names of the victim rigid objects.

    Returns:
        Victim positions in the world frame. Shape ``(num_envs, num_victims, 3)``.
    """
    return torch.stack([cast_victim(env, name).data.root_pos_w.torch for name in victim_names], dim=1)


def cast_victim(env: ManagerBasedEnv, name: str) -> RigidObject:
    """Fetch a victim prop from the scene (separate helper purely for type-hinting)."""
    return env.scene[name]


def _nearest_victim(
    env: ManagerBasedRLEnv,
    victim_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planar distance and bearing error to the nearest victim.

    Returns:
        ``(distance, heading_error)``. ``distance`` is the XY distance [m] to the nearest
        victim, shape ``(num_envs,)``. ``heading_error`` is the wrapped angle [rad] between
        the robot's forward axis and the direction to that victim, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w.torch  # (N, 3)
    victims_w = _victim_positions_w(env, victim_names)  # (N, V, 3)

    delta_xy = victims_w[..., :2] - root_pos_w[:, None, :2]  # (N, V, 2)
    dist = torch.norm(delta_xy, dim=-1)  # (N, V)
    nearest_dist, nearest_idx = dist.min(dim=1)  # (N,), (N,)

    nearest_delta = delta_xy.gather(1, nearest_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)  # (N, 2)
    target_yaw = torch.atan2(nearest_delta[:, 1], nearest_delta[:, 0])
    heading_err = math_utils.wrap_to_pi(target_yaw - asset.data.heading_w.torch)
    return nearest_dist, heading_err


##
# Observations
##


def lidar_ranges(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_range: float = 10.0,
) -> torch.Tensor:
    """Per-ray range measurements from a ray-cast LiDAR, normalized to ``[0, 1]``.

    Misses (``inf`` hit positions, see :attr:`RayCasterData.ray_hits_w`) are reported as the
    maximum range.

    Args:
        env: The environment.
        sensor_cfg: The ray-caster sensor to read.
        max_range: Range at which the reading saturates [m]. Should match the sensor's
            ``max_distance``.

    Returns:
        Normalized ranges. Shape ``(num_envs, num_rays)``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_w = sensor.data.ray_hits_w.torch  # (N, B, 3)
    origin_w = sensor.data.pos_w.torch.unsqueeze(1)  # (N, 1, 3)
    ranges = torch.norm(hits_w - origin_w, dim=-1)
    ranges = torch.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=max_range)
    return (ranges / max_range).clamp(0.0, 1.0)


def thermal_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    flatten: bool = True,
) -> torch.Tensor:
    """Pseudo-thermal image synthesized from the camera's semantic segmentation output.

    Args:
        env: The environment.
        sensor_cfg: The camera sensor. Must have ``"semantic_segmentation"`` in
            ``data_types`` and colorized segmentation enabled.
        flatten: If True, flatten to ``(num_envs, H * W)``. If False, keep
            ``(num_envs, H, W, 1)``.

    Returns:
        Normalized temperature image in ``[0, 1]``.
    """
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    # ``CameraData.output`` maps data type -> ProxyArray; ``.torch`` gives a zero-copy view.
    seg = sensor.data.output["semantic_segmentation"].torch  # (N, H, W, 4) uint8
    # The buffer's layout depends on the renderer's colorization setting: (N,H,W,4) uint8
    # when colorized, (N,H,W,1) int32 raw. The raw layout does not raise here -- seg[..., :3]
    # would broadcast a single channel to three and hand back a plausible-looking image of
    # pure garbage. Fail loudly instead; this path has never run on GPU.
    assert seg.shape[-1] == 4 and seg.dtype == torch.uint8, (
        f"expected colorized segmentation (N,H,W,4) uint8, got {tuple(seg.shape)} {seg.dtype}; "
        "check colorize_semantic_segmentation on the renderer cfg"
    )
    temperature = thermal.segmentation_to_temperature(seg)  # (N, H, W)
    normalized = thermal.normalize_temperature(temperature)
    if flatten:
        return normalized.reshape(env.num_envs, -1)
    return normalized.unsqueeze(-1)


def victim_relative_pos_b(
    env: ManagerBasedRLEnv,
    victim_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the nearest victim expressed in the robot base frame.

    Intended as a privileged / critic observation, not something a real robot would have.

    Returns:
        Shape ``(num_envs, 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w.torch
    root_quat_w = asset.data.root_quat_w.torch
    victims_w = _victim_positions_w(env, victim_names)

    delta_w = victims_w - root_pos_w[:, None, :]
    nearest_idx = torch.norm(delta_w[..., :2], dim=-1).argmin(dim=1)
    nearest_delta_w = delta_w.gather(1, nearest_idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
    return math_utils.quat_apply_inverse(root_quat_w, nearest_delta_w)


##
# Rewards
##


def victim_proximity_exp(
    env: ManagerBasedRLEnv,
    std: float,
    victim_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward that increases monotonically as the robot gets closer to the nearest victim.

    ``exp(-d / std)`` where ``d`` is the planar distance [m] to the nearest victim. Any
    action that reduces ``d`` increases this reward.

    Returns:
        Shape ``(num_envs,)``.
    """
    dist, _ = _nearest_victim(env, victim_names, asset_cfg)
    return torch.exp(-dist / std)


def victim_heading_exp(
    env: ManagerBasedRLEnv,
    std: float,
    victim_names: tuple[str, ...],
    max_distance: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for pointing the robot's forward axis at the nearest victim.

    ``exp(-|heading_error| / std)``, gated to zero beyond ``max_distance`` so the robot is
    not asked to stare at a victim it cannot perceive.

    Returns:
        Shape ``(num_envs,)``.
    """
    dist, heading_err = _nearest_victim(env, victim_names, asset_cfg)
    reward = torch.exp(-heading_err.abs() / std)
    return reward * (dist < max_distance).float()

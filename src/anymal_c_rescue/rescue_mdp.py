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
from isaaclab.managers import SceneEntityCfg

from . import thermal

# Articulation / RigidObject / Camera / RayCaster are the *runtime* classes, and importing
# them at module scope crashes the process outright -- exit 134/139 before a single line of
# our code runs. They can only be imported once SimulationApp exists, and this module is
# imported well before that: hydra resolves ``env_cfg_entry_point`` (rescue_env_cfg, which
# imports this file) inside ``_run_hydra``, which runs before ``launch_simulation``.
# Isaac Lab's own mdp modules (e.g. velocity/mdp/rewards.py) guard them the same way.
#
# Safe because every use below is an annotation only, and ``from __future__ import
# annotations`` above makes annotations strings that are never evaluated at runtime. If you
# ever need one of these at runtime, import it inside the function body, not here.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import Camera, RayCaster


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


##
# Search task -- placement and rewards
##


def reset_victims_annulus(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    victim_names: tuple[str, ...],
    radius_range: tuple[float, float] = (1.6, 3.6),
    drop_height: float = 2.0,
) -> None:
    """Scatter the victims to fresh positions around each environment origin on reset.

    This is what turns the task into an actual search. With victims pinned to fixed offsets
    the optimal policy is "walk 2.9 m along a fixed bearing", learnable from proprioception
    alone -- the LiDAR and the thermal camera are then pure cost with no influence on return.
    Re-drawing the layout every episode makes that memorised bearing worthless.

    Each victim is drawn in its own angular sector (victim ``i`` of ``V`` gets
    ``[i, i+1) * 2*pi/V``), which guarantees they cannot land on top of each other without
    rejection sampling. Radius is sampled ``sqrt``-weighted so positions are uniform over
    *area* rather than clustered near the centre.

    Args:
        env: The environment.
        env_ids: Environments being reset.
        victim_names: Scene entity names of the victim rigid objects.
        radius_range: Min/max planar distance from the environment origin [m]. Keep the max
            inside the 8x8 m terrain patch half-width of 4.0 m.
        drop_height: Spawn height above the patch origin [m]. See ``VICTIM_OFFSETS`` for why
            this is generous rather than snug.
    """
    num = len(env_ids)
    if num == 0:
        return
    origins = env.scene.env_origins[env_ids]  # (M, 3)
    n_victims = len(victim_names)
    sector = 2.0 * torch.pi / n_victims
    r_min, r_max = radius_range

    for i, name in enumerate(victim_names):
        victim = cast_victim(env, name)
        angle = (torch.rand(num, device=origins.device) + i) * sector
        # sqrt-weighted radius -> uniform over the annulus area, not biased inward
        u = torch.rand(num, device=origins.device)
        radius = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)

        pos = origins.clone()
        pos[:, 0] += radius * torch.cos(angle)
        pos[:, 1] += radius * torch.sin(angle)
        pos[:, 2] += drop_height

        yaw = (torch.rand(num, device=origins.device) * 2.0 - 1.0) * torch.pi
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )

        victim.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        victim.write_root_velocity_to_sim(
            torch.zeros((num, 6), device=origins.device), env_ids=env_ids
        )


def _detectable_victims(
    env: ManagerBasedRLEnv,
    victim_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    detect_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest victim that is close enough to be sensed, plus the unit vector to it.

    ``detect_range`` should track the LiDAR's max range: beyond it the robot genuinely has
    no way of knowing a victim is there, so paying it to walk that way would be rewarding
    clairvoyance rather than sensing.

    Returns:
        ``(has_target, distance, unit_xy)``. ``has_target`` is a bool mask of environments
        with at least one victim in range, shape ``(num_envs,)``. ``distance`` is the planar
        distance to the nearest such victim (``inf`` where none), shape ``(num_envs,)``.
        ``unit_xy`` is the unit vector toward it, shape ``(num_envs, 2)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w.torch
    victims_w = _victim_positions_w(env, victim_names)

    delta_xy = victims_w[..., :2] - root_pos_w[:, None, :2]  # (N, V, 2)
    dist = torch.norm(delta_xy, dim=-1)  # (N, V)

    out_of_range = dist > detect_range
    dist_masked = dist.masked_fill(out_of_range, float("inf"))
    nearest_dist, nearest_idx = dist_masked.min(dim=1)
    has_target = torch.isfinite(nearest_dist)

    nearest_delta = delta_xy.gather(1, nearest_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
    unit_xy = nearest_delta / nearest_delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return has_target, nearest_dist, unit_xy


def victim_approach_rate(
    env: ManagerBasedRLEnv,
    victim_names: tuple[str, ...],
    detect_range: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Speed at which the robot is closing on the nearest detectable victim [m/s].

    Replaces the old ``victim_proximity_exp``. That term paid ``exp(-d/std)`` for *being*
    near a victim, which is maximised by arriving and standing still -- and standing still
    is exactly what ``terrain_levels_vel`` demotes, so it collapsed the terrain curriculum
    (measured: levels fell 0.68 -> 0.38 over 100 iterations against a baseline of ~5.3).

    This pays for *closing distance* instead. It is stateless -- just the base velocity
    projected onto the direction to the target -- so standing still pays exactly zero and
    there is nothing to farm. Negative values are clipped so retreating is merely worthless
    rather than punished, which keeps it from fighting the velocity command tracking.

    Returns zero where no victim is within ``detect_range``, so the robot is never paid to
    walk toward something it could not possibly have sensed. That is what makes the LiDAR
    and thermal camera load-bearing: to earn this reward the policy must first find a
    victim, and the sensors are the only way to do that.

    Returns:
        Closing speed [m/s], clipped at zero. Shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    has_target, _, unit_xy = _detectable_victims(env, victim_names, asset_cfg, detect_range)
    vel_xy = asset.data.root_lin_vel_w.torch[:, :2]
    rate = (vel_xy * unit_xy).sum(dim=-1).clamp(min=0.0)
    return torch.where(has_target, rate, torch.zeros_like(rate))


def victim_found(
    env: ManagerBasedRLEnv,
    victim_names: tuple[str, ...],
    found_radius: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse per-step bonus for standing within ``found_radius`` of a victim.

    This is the actual objective -- reaching a casualty. It is deliberately sparse and
    small-radius so it cannot be farmed from a distance the way a smooth proximity kernel
    can; the dense guidance comes from :func:`victim_approach_rate` instead.

    Returns:
        1.0 where a victim is within ``found_radius``, else 0.0. Shape ``(num_envs,)``.
    """
    dist, _ = _nearest_victim(env, victim_names, asset_cfg)
    return (dist < found_radius).float()

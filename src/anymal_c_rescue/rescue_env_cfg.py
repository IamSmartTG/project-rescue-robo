# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 sensor-equipped rescue environment for ANYmal-C.

Derived from ``Isaac-Velocity-Rough-Anymal-C-v0``. Adds, on top of the unmodified
locomotion task:

* a ray-cast LiDAR on the trunk (``MultiMeshRayCasterCfg`` + ``LidarPatternCfg``),
* an emergency-supply payload (randomized added mass + added inertia on the trunk),
* a pseudo-thermal camera (``CameraCfg`` producing ``semantic_segmentation``, converted
  to temperatures by :mod:`.thermal`),
* static victim props carrying the ``class:victim`` semantic tag,
* victim-search reward terms (proximity + orientation),
* observation terms exposing the LiDAR ranges and the thermal image.

Launch note: the thermal camera requires ``--enable_cameras``. In Isaac Sim 6.0 use
``--viz none`` for force-headless (``--headless`` is deprecated).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.sim.schemas.schemas_cfg import CollisionBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventsCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
)

from . import rescue_mdp
from . import thermal as thermal_utils

##
# Constants
##

VICTIM_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (2.60, 1.40, 2.00),
    (-2.10, 2.55, 2.00),
    (0.60, -2.80, 2.00),
)
"""Victim prop spawn positions as (x, y, z) offsets from each environment origin [m].

Kept inside the 8 x 8 m terrain patch produced by ``ROUGH_TERRAINS_CFG``. The z value is a
drop height: the props are dynamic rigid bodies and settle onto the uneven terrain.

.. important::
    z is 2.0, not a snug 0.6, because ``env_origins`` is the patch **centre** height while
    the victims sit ~2.9-3.3 m out from it. ``ROUGH_TERRAINS_CFG`` includes pyramid stairs
    and slopes whose relief reaches 1-2 m at that radius, so a 0.6 m drop spawned the box
    *inside* the terrain on inverted patches -- interpenetration, which the solver answers
    by ejecting a 45 kg body. Dropping from clear above costs a 0.64 s fall at episode
    start. The exact fix is to sample terrain height at the drawn xy, which needs a custom
    event term; not worth it until this is seen to be insufficient.

.. important::
    The props are :class:`RigidObjectCfg`, *not* static :class:`AssetBaseCfg` prims, and each
    one gets a ``reset`` event that writes ``default_root_pose + env_origins``. This is
    required for correctness: ``InteractiveScene.clone_environments`` places the ``env_*``
    USD prims on a regular ``env_spacing`` grid
    (``_default_env_pose = cloner.grid_transforms(...)``), whereas ``scene.env_origins``
    comes from the terrain generator's patch centres. A static prim authored under
    ``{ENV_REGEX_NS}`` would therefore sit at the grid position, nowhere near the robot,
    which is repositioned every reset by ``reset_root_state_uniform``.
"""

LIDAR_MAX_RANGE: float = 10.0
"""LiDAR saturation range [m]. Must match the sensor ``max_distance`` and the observation."""

THERMAL_RESOLUTION: tuple[int, int] = (64, 64)
"""(width, height) of the pseudo-thermal camera. Small on purpose, for training throughput."""

VICTIM_ASSET_NAMES: tuple[str, ...] = ("victim_0", "victim_1", "victim_2")
"""Scene entity names of the victim props, used by the victim-search reward terms."""


def _victim_asset_cfg(index: int, offset: tuple[float, float, float]) -> RigidObjectCfg:
    """Build one victim prop.

    A cuboid (not a capsule -- it must not roll down a slope) spawned as a
    ``UsdGeom.Mesh``, so it is both ray-castable by the LiDAR and taggable for the thermal
    camera. Heavy enough (45 kg) that the robot cannot fling it around.

    ``RigidBodyBaseCfg`` / ``CollisionBaseCfg`` are the solver-common base schemas; the
    more commonly seen ``sim_utils.RigidBodyPropertiesCfg`` /
    ``sim_utils.CollisionPropertiesCfg`` names are PhysX-specific in 6.0 and only resolve
    through a deprecated forwarding shim in ``isaaclab/sim/__init__.py``.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Victim_{index}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=offset),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.70, 0.35, 0.28),
            rigid_props=sim_utils.RigidBodyBaseCfg(),
            collision_props=CollisionBaseCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            semantic_tags=[("class", "victim")],
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.35, 0.25), emissive_color=(0.25, 0.05, 0.0), roughness=0.8
            ),
        ),
    )


##
# Scene
##


@configclass
class RescueSceneCfg(MySceneCfg):
    """Rough-terrain scene plus rescue sensors and victim props."""

    # -- victim props (semantically tagged rigid bodies)
    victim_0: RigidObjectCfg = _victim_asset_cfg(0, VICTIM_OFFSETS[0])
    victim_1: RigidObjectCfg = _victim_asset_cfg(1, VICTIM_OFFSETS[1])
    victim_2: RigidObjectCfg = _victim_asset_cfg(2, VICTIM_OFFSETS[2])

    # -- ray-cast LiDAR on the trunk
    # MultiMeshRayCaster (not RayCaster) because the plain RayCaster hard-errors on more
    # than one mesh prim path, and the LiDAR must see both the ground and the victims.
    # ray_alignment="base": in "yaw" mode the kernel deliberately does NOT rotate ray
    # DIRECTIONS, which is correct for a downward height scanner but wrong for a LiDAR.
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.18)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            # The fan is tilted DOWN, not centred on the horizon. Sensor height is
            # ~0.75 m (base ~0.57 + 0.18 offset) and a victim is a 0.28 m box, so a
            # symmetric (-15, +15) fan put its two upper beams permanently in the sky and
            # left a blind band from 2.80 m (where the -15 beam hits ground) out to
            # 5.37 m (where the -5 beam first clears the box top) -- exactly the 2.86-3.30 m
            # band the victims spawn in. (-25, +5) over 8 channels gives ~4.3 deg spacing
            # and continuous coverage from ~1.0 m to beyond LIDAR_MAX_RANGE.
            channels=8,
            vertical_fov_range=(-25.0, 5.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10.0,
        ),  # -> 8 x 36 = 288 rays (the 360-deg wrap-around ray is dropped by lidar_pattern)
        max_distance=LIDAR_MAX_RANGE,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            # track_mesh_transforms=True is mandatory here, not an optimization choice: the
            # victims are repositioned from the authored clone-grid pose to
            # ``env_origins + offset`` at reset, so a frozen mesh transform would raycast
            # against phantom geometry. Requires RigidBodyAPI on an ancestor prim, which
            # holds because the victims are RigidObjectCfg.
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Victim_.*", track_mesh_transforms=True
            ),
        ],
        debug_vis=False,
    )

    # -- pseudo-thermal camera on the trunk
    # convention="world" means forward = +X and up = +Z, so identity rotation points the
    # camera straight out of the robot's nose. Requires --enable_cameras at launch.
    thermal_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/ThermalCamera",
        offset=CameraCfg.OffsetCfg(pos=(0.35, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["semantic_segmentation"],
        width=THERMAL_RESOLUTION[0],
        height=THERMAL_RESOLUTION[1],
        # Far clip is 8.0, not 20.0: terrain patches are 8x8 m on one continuous ground
        # mesh, so a 20 m far plane rendered ~2.5 patches in every direction and filled the
        # segmentation buffer with neighbouring envs' robots and victims -- heat sources
        # this env is never rewarded for reaching.
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 8.0)),
        renderer_cfg=IsaacRtxRendererCfg(
            semantic_filter=thermal_utils.SEMANTIC_FILTER,
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=thermal_utils.SEMANTIC_SEGMENTATION_MAPPING,
        ),
        debug_vis=False,
    )


##
# MDP
##


@configclass
class RescueObservationsCfg(ObservationsCfg):
    """Locomotion observations plus LiDAR ranges and the thermal image."""

    @configclass
    class RescuePolicyCfg(ObservationsCfg.PolicyCfg):
        """Proprioception + height scan (inherited) + LiDAR."""

        lidar_ranges = ObsTerm(
            func=rescue_mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_range": LIDAR_MAX_RANGE},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 1.0),
        )

    @configclass
    class ThermalCfg(ObsGroup):
        """Flattened pseudo-thermal image, kept out of the proprioceptive group."""

        thermal = ObsTerm(
            func=rescue_mdp.thermal_image,
            params={"sensor_cfg": SceneEntityCfg("thermal_camera"), "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: RescuePolicyCfg = RescuePolicyCfg()
    thermal: ThermalCfg = ThermalCfg()


@configclass
class RescueEventsCfg(EventsCfg):
    """Locomotion events plus the emergency-supply payload."""

    # Payload mass. Anymal-C's trunk is ~30 kg, so +2..12 kg is a 7-40% change and
    # meaningfully alters gait. ``recompute_inertia=True`` scales the trunk inertia with
    # the new mass (verified default in ``randomize_rigid_body_mass.__call__``).
    add_payload_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (2.0, 12.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # Extra rotational inertia from a payload that is not a point mass at the trunk COM.
    # Declared after the mass term so it runs after it -- ``randomize_rigid_body_inertia``
    # snapshots its default inertia on first call, which must be the post-mass value.
    add_payload_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "inertia_distribution_params": (0.02, 0.30),
            "operation": "add",
            "distribution": "uniform",
            "diagonal_only": True,
        },
    )

    # Move the victim props from their authored clone-grid pose onto this env's terrain
    # patch. ``reset_root_state_uniform`` writes ``default_root_pose + env_origins``.
    reset_victim_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # x/y jitter is required for correctness, not variety. ROUGH_TERRAINS_CFG is
            # 10 rows x 20 cols = 200 patches, so at num_envs=4096 roughly 20 envs share
            # one env_origins row. With yaw-only randomization every one of those envs
            # wrote its victim to the identical world position -- ~20 coincident 45 kg
            # cuboids with a degenerate contact normal, re-stacked on every reset.
            # +-1.0 m keeps all three inside the 8x8 m patch (worst case victim_2 reaches
            # y = -3.8 against a 4.0 m half-width).
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.1416, 3.1416)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("victim_0"),
        },
    )
    reset_victim_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # x/y jitter is required for correctness, not variety. ROUGH_TERRAINS_CFG is
            # 10 rows x 20 cols = 200 patches, so at num_envs=4096 roughly 20 envs share
            # one env_origins row. With yaw-only randomization every one of those envs
            # wrote its victim to the identical world position -- ~20 coincident 45 kg
            # cuboids with a degenerate contact normal, re-stacked on every reset.
            # +-1.0 m keeps all three inside the 8x8 m patch (worst case victim_2 reaches
            # y = -3.8 against a 4.0 m half-width).
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.1416, 3.1416)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("victim_1"),
        },
    )
    reset_victim_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # x/y jitter is required for correctness, not variety. ROUGH_TERRAINS_CFG is
            # 10 rows x 20 cols = 200 patches, so at num_envs=4096 roughly 20 envs share
            # one env_origins row. With yaw-only randomization every one of those envs
            # wrote its victim to the identical world position -- ~20 coincident 45 kg
            # cuboids with a degenerate contact normal, re-stacked on every reset.
            # +-1.0 m keeps all three inside the 8x8 m patch (worst case victim_2 reaches
            # y = -3.8 against a 4.0 m half-width).
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.1416, 3.1416)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("victim_2"),
        },
    )


@configclass
class RescueRewardsCfg(RewardsCfg):
    """All locomotion reward terms (unchanged) plus victim-search shaping."""

    # -- victim search
    victim_proximity = RewTerm(
        func=rescue_mdp.victim_proximity_exp,
        weight=1.0,
        params={"std": 2.0, "victim_names": VICTIM_ASSET_NAMES},
    )
    victim_heading = RewTerm(
        func=rescue_mdp.victim_heading_exp,
        weight=0.5,
        params={"std": 0.8, "victim_names": VICTIM_ASSET_NAMES, "max_distance": 6.0},
    )


##
# Environment
##


@configclass
class AnymalCRescueEnvCfg(AnymalCRoughEnvCfg):
    """Sensor-equipped rescue variant of the ANYmal-C rough locomotion task."""

    scene: RescueSceneCfg = RescueSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RescueObservationsCfg = RescueObservationsCfg()
    events: RescueEventsCfg = RescueEventsCfg()
    rewards: RescueRewardsCfg = RescueRewardsCfg()

    def __post_init__(self):
        # runs AnymalCRoughEnvCfg.__post_init__ -> LocomotionVelocityRoughEnvCfg.__post_init__
        super().__post_init__()

        # tag the robot so the thermal camera resolves it as a warm (but not victim) class
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # The inherited ``add_base_mass`` term also randomizes the trunk mass. Two
        # ``randomize_rigid_body_mass`` terms on the same body fight each other: each one
        # snapshots ``asset.data.body_mass`` on its own first call and then resets to that
        # snapshot before applying, so the result is order-dependent. The payload term
        # (uniform +2..12 kg) subsumes it, so the inherited term is disabled.
        self.events.add_base_mass = None

        # tick the new sensors at the policy rate
        self.scene.lidar.update_period = self.decimation * self.sim.dt
        self.scene.thermal_camera.update_period = self.decimation * self.sim.dt


@configclass
class AnymalCRescueEnvCfg_PLAY(AnymalCRescueEnvCfg):
    """Small, deterministic variant for inspection."""

    def __post_init__(self):
        super().__post_init__()

        # smaller scene
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # show the LiDAR hits
        self.scene.lidar.debug_vis = True


@configclass
class AnymalCRescueNoCamEnvCfg(AnymalCRescueEnvCfg):
    """Rescue task with the pseudo-thermal camera removed.

    This exists because of a pod limitation, not a design choice: the RTX renderer cannot
    start on the current RunPod container (``vkCreateInstance`` fails -- the NVIDIA Vulkan
    ICD loads and exports its entry point, but reports no usable device, while CUDA compute
    is entirely healthy). ``--enable_cameras`` therefore aborts the process at launch.

    LiDAR is warp raycasting on CUDA and needs no Vulkan, so this variant is the only way to
    exercise the victim props, the corrected LiDAR geometry and the search rewards on GPU
    while that is unresolved. Delete it once RTX rendering works.
    """

    def __post_init__(self):
        super().__post_init__()
        # Drop the camera itself, its observation group, and the update_period the parent
        # set on it. Order matters: the parent's __post_init__ touches thermal_camera.
        self.scene.thermal_camera = None
        self.observations.thermal = None

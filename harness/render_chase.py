"""Render a chase-cam demo of a trained policy.

Why this exists: Isaac Lab's ``--video`` records the *viewport* camera, which is static.
A walking robot leaves frame within seconds, so the stock clips show empty terrain for most
of their length. ``ViewerCfg.origin_type="asset_root"`` would track the robot but is
deprecated in 6.0 and silently ignored, and overriding ``env.viewer.*`` through hydra just
aimed the static camera somewhere else.

The fix is to stop using the viewport at all: parent an RGB ``CameraCfg`` to the robot's own
base prim, exactly like the thermal sensor. A camera bolted to the robot cannot lose it.

Usage:
    ./isaaclab.sh -p harness/render_chase.py --task Isaac-Rescue-Anymal-C-NoCam-v0 \
        --checkpoint ~/rescue_policy.pt --num_envs 1 --steps 900 --out ~/chase
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Rescue-Anymal-C-NoCam-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--out", type=str, default="chase")
# Which episode you get is pure luck of the draw, and a demo of the robot faceplanting at
# 3 s is not a demo. The driving script re-rolls this until it gets a usable take.
parser.add_argument("--seed", type=int, default=42)
# Spawn the casualty much further out than training does, so the video shows a search rather
# than a two-second walk. Safe here and not in training: this renders ONE robot, so there is
# no neighbouring 8x8 m patch to contaminate -- which is the only reason the training annulus
# is capped at 3.8 m.
#
# 6.5 m is near the real ceiling, and the ceiling is the sensor, not the terrain. The thermal
# camera's far clip is 8.0 m, so past that the casualty is not faint, it is absent, and the
# robot is wandering blind until it stumbles into range. That films worse than an easy find,
# not better. A ~20 s episode also has to cover scanning plus the approach.
parser.add_argument(
    "--victim_radius",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="override the victim spawn annulus [m], e.g. --victim_radius 5.0 6.5",
)
# Opt-in, not default. The only render made with it produced a black chase panel and a blank
# thermal panel -- see the long note at the thermal_viz camera. Verify a frame before trusting
# a video made with this on.
# Without this the renderer runs only the frozen walker. On the Nav task that means the
# residual delta stays at its initialised zeros, so the video shows the SCRIPTED SEEKER while
# a caption claims a trained navigator -- a plausible-looking artifact that means the opposite
# of what it says. Pass the navigator checkpoint to actually film the trained policy.
parser.add_argument("--navigator_checkpoint", type=str, default=None)
# Films the no-rubble scene, which is measurably easier: at n=1024 removing the debris moves
# success 0.1602 -> 0.2266 and the fall rate 0.525 -> 0.275. A take from here shows a more
# capable robot than any number in the write-up describes, so anything rendered with this must
# be labelled as the no-debris scene. summary.json records the flag for that reason.
parser.add_argument(
    "--no_debris",
    action="store_true",
    help="banish the rubble; renders the easier no-debris scene -- label any output",
)
parser.add_argument("--inner_steps", type=int, default=5, help="physics steps per navigator decision")
parser.add_argument(
    "--thermal_viz",
    action="store_true",
    help="add a 64x64 viz-only thermal camera for the panel (UNVERIFIED -- see notes)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # the RTX renderer must be up for any camera to produce pixels

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imported only after the app exists -- Isaac Lab runtime classes segfault if imported
# before SimulationApp (cost us an exit-139 earlier in this project).
import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

os.makedirs(args.out, exist_ok=True)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.seed = args.seed

# Widen the search before the scene is built. Mutating the event term's params is the whole
# change -- reset_victims_annulus already takes radius_range, so nothing new is needed.
#
# Note this is deliberately out of distribution: the policy trains at 2.5-3.8 m and is being
# filmed at up to 6.5. The behaviour required is the same one it learned (turn, scan, walk at
# the warm blob), just more of it, and the take-selection loop re-rolls the seed until it
# finds a clean episode -- so a harder setup costs re-rolls, not the deliverable. Say so in
# the write-up rather than implying the policy was trained for this range.
if args.no_debris:
    # Same mechanism as eval_rescue.py --no_debris: move the blocks out of reach rather than
    # deleting the assets, so the scene the LiDAR raycasts against is otherwise unchanged.
    # Measured cost of the rubble at n=1024: success 0.1602 -> 0.2266, falls 0.525 -> 0.275.
    if getattr(env_cfg.events, "reset_debris", None) is not None:
        env_cfg.events.reset_debris.params["radius_range"] = (60.0, 62.0)
        print("[INFO] debris banished -- THIS RENDER IS THE NO-RUBBLE SCENE, label it as such")
    else:
        print("[WARN] --no_debris given but this task has no reset_debris event")

if args.victim_radius is not None:
    _reset_victims = getattr(env_cfg.events, "reset_victims", None)
    if _reset_victims is None:
        raise SystemExit("--victim_radius given but this task has no reset_victims event")
    _r = (float(args.victim_radius[0]), float(args.victim_radius[1]))
    if _r[0] >= _r[1]:
        raise SystemExit(f"--victim_radius needs MIN < MAX, got {_r}")
    # Past the thermal far clip the casualty simply is not in the image, so the search
    # becomes a random walk. Warn rather than refuse -- it is a legitimate thing to film
    # deliberately, just not by accident.
    _far_clip = getattr(env_cfg.scene, "thermal_camera", None)
    if _far_clip is not None and _r[1] > _far_clip.spawn.clipping_range[1]:
        print(
            f"[WARN] max radius {_r[1]} m exceeds the thermal far clip "
            f"{_far_clip.spawn.clipping_range[1]} m -- the casualty will be invisible to the "
            "sensor at spawn and the robot can only find it by wandering into range"
        )
    _reset_victims.params["radius_range"] = _r
    print(f"[INFO] victim spawn annulus overridden to {_r[0]}-{_r[1]} m (training uses 2.5-3.8)")

# Chase camera: 2.6 m behind, 1.3 m up, pitched ~18 deg down.
# convention="world" means forward is +X, so a negative x offset puts the camera behind the
# robot still looking forward along +X -- i.e. over its shoulder. rot is (x, y, z, w); a
# rotation about +Y tips forward +X down toward -Z.
_PITCH_DEG = 18.0
_half = np.deg2rad(_PITCH_DEG) / 2.0
env_cfg.scene.chase_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/ChaseCam",
    offset=CameraCfg.OffsetCfg(
        pos=(-2.6, 0.0, 1.3),
        rot=(0.0, float(np.sin(_half)), 0.0, float(np.cos(_half))),
        convention="world",
    ),
    data_types=["rgb"],
    width=1280,
    height=720,
    spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 80.0)),
    update_period=0.0,  # every physics-render tick, so no frames are skipped
    # Explicit default renderer. Without it this camera renders pure black whenever the
    # scene also contains the pseudo-thermal camera, which carries its own
    # IsaacRtxRendererCfg(semantic_filter="class:*"). It renders fine in the camera-free
    # task. Giving each camera its own renderer config stops them sharing one.
    renderer_cfg=IsaacRtxRendererCfg(),
)

# The chase camera renders pure black in any task that also carries the thermal camera,
# while working fine in the camera-free task. Per-camera IsaacRtxRendererCfg did not fix it,
# so the remaining theory is that the RTX renderer only produces a colour AOV when some
# camera asks for one -- and the thermal camera asks only for semantic_segmentation.
# Requesting rgb there too is the cheap test of that.
_thermal_cfg = getattr(env_cfg.scene, "thermal_camera", None)
if _thermal_cfg is not None and "rgb" not in _thermal_cfg.data_types:
    _thermal_cfg.data_types = list(_thermal_cfg.data_types) + ["rgb"]
    print("[INFO] added 'rgb' to thermal camera data_types (black-panel workaround test)")

# A second, higher-resolution thermal camera that exists only to be looked at.
#
# The policy's own thermal image is 16x16 and has to stay that way: at 64x64 the flattened
# image is 4096 dims against ~523 of everything else, which collapsed training outright
# (rescue_env_cfg.py THERMAL_RESOLUTION records the measurement -- episode length 19-24 ->
# 7.76 steps, 100% base-contact). But a 16x16 buffer blown up to a 360 px panel reads as a
# broken sensor rather than as a thermal camera, and nothing requires the *video* to display
# the actor's own buffer.
#
# Safe because it is a scene sensor and nothing more: no observation term names "thermal_viz",
# so it never reaches the observation manager and the checkpoint's input width is untouched.
# Same mount, lens and clipping as the real sensor, so the view is the policy's view -- just
# sampled finer. Deliberately semantic_segmentation only: this camera is never asked for an
# rgb AOV, since the chase camera already forces one.
_THERMAL_VIZ_RESOLUTION = (64, 64)
# OFF BY DEFAULT, because the one render made with it came out unusable.
#
# rescue_demo_v4.mp4: chase camera pure black and the thermal panel uniform ambient with no
# casualty blob, while rescue_demo_v3.mp4 -- same code path, previous host, no thermal_viz --
# rendered correctly. The LiDAR panel was unaffected, which fits: it is warp raycasting and
# never touches the RTX renderer.
#
# Unproven but most likely: this adds a *third* RTX camera carrying a deepcopied
# IsaacRtxRendererCfg and an extra distance_to_image_plane AOV, and the comment at the chase
# camera above records that this renderer is already sensitive to exactly that -- which
# cameras ask for which AOVs. The confounder is that the same run was also the first on
# Blackwell and Ubuntu 24.04, and the instance was terminated before either could be ruled
# out.
#
# Until someone reproduces it, the default path is the one known to produce a watchable
# video. Pass --thermal_viz to opt in, and check the chase panel is not black before
# trusting the output.
if _thermal_cfg is not None and args.thermal_viz:
    # deepcopy, not a shared reference, for offset/spawn/renderer_cfg. The chase camera above
    # is commented with the lesson: two cameras sharing one IsaacRtxRendererCfg is exactly how
    # the black-panel bug behaved, and "give each camera its own renderer config" was the
    # response. Passing _thermal_cfg.renderer_cfg straight through would hand the viz camera
    # the *same object* the policy's camera holds and walk back into that. Same reasoning for
    # spawn: PinholeCameraCfg is consumed to author a prim, and two prims sharing one spawn
    # cfg is a mutation hazard for no benefit.
    import copy  # noqa: E402

    env_cfg.scene.thermal_viz = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/ThermalViz",
        offset=copy.deepcopy(_thermal_cfg.offset),
        # distance_to_image_plane as well as the segmentation: the radiometry below needs a
        # per-pixel range to attenuate against, and the policy's camera does not render one.
        data_types=["semantic_segmentation", "distance_to_image_plane"],
        width=_THERMAL_VIZ_RESOLUTION[0],
        height=_THERMAL_VIZ_RESOLUTION[1],
        spawn=copy.deepcopy(_thermal_cfg.spawn),
        # Every frame, unlike the policy's camera, which ticks at a quarter rate to save
        # training throughput. A panel that updates once every four frames looks stuttery,
        # and there is no throughput budget to protect in a single-robot render.
        update_period=0.0,
        renderer_cfg=copy.deepcopy(_thermal_cfg.renderer_cfg),
    )
    print(f"[INFO] added {_THERMAL_VIZ_RESOLUTION[0]}x{_THERMAL_VIZ_RESOLUTION[1]} viz-only thermal camera")

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
# Same shim play.py applies. Without it the runner passes a 'stochastic' kwarg that this
# rsl-rl version's MLPModel does not accept, and construction dies with a TypeError.
from importlib import metadata  # noqa: E402

agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[INFO] loaded checkpoint: {args.checkpoint}")

# Optional high-level navigator. Reuses NavigatorEnv purely as the adapter that tells rsl-rl
# the action space is 3 (a velocity command) rather than 12 (joint targets) -- without it the
# runner sizes the actor from the env and rebuilds the wrong network shape.
_nav_policy = None
_nav_cmd = None
if args.navigator_checkpoint:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c_rescue.nav_env import (  # noqa: E402
        NavigatorEnv,
    )

    _nav_cmd = env.unwrapped.command_manager.get_term("base_velocity")
    if not hasattr(_nav_cmd, "set_delta"):
        raise SystemExit(
            f"--navigator_checkpoint given but base_velocity is {type(_nav_cmd).__name__}. "
            "Use the -Nav-Play- task; the plain PLAY task keeps the random velocity command."
        )
    _nav_shim = NavigatorEnv(env, policy, _nav_cmd, args.inner_steps)
    _nav_cfg = load_cfg_from_registry(args.task, "navigator_cfg_entry_point")
    _nav_cfg = handle_deprecated_rsl_rl_cfg(_nav_cfg, metadata.version("rsl-rl-lib"))
    _nav_runner = OnPolicyRunner(_nav_shim, _nav_cfg.to_dict(), log_dir=None, device=args.device)
    _nav_runner.load(args.navigator_checkpoint)
    _nav_policy = _nav_runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] navigator loaded: {args.navigator_checkpoint} (delta every {args.inner_steps} steps)")
else:
    print("[INFO] no navigator -- command comes from the scripted seeker alone")

# rsl-rl 5.x returns a single obs object here, not an (obs, extras) tuple.
obs = env.get_observations()
cam = env.unwrapped.scene.sensors["chase_camera"]

# If the task carries the pseudo-thermal sensor, composite its view beside the chase view:
# left = what an observer sees, right = what the robot senses. That pairing is the whole
# argument that the sensor is doing something, and it is far more legible than stills.
thermal_cam = env.unwrapped.scene.sensors.get("thermal_camera")
_thermal_utils = None
if thermal_cam is not None:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c_rescue import (  # noqa: E402
        thermal as _thermal_utils,
    )
    print("[INFO] thermal camera present -- rendering side-by-side composite")

# Draw the panel from the viz camera when it exists, else fall back to the policy's own
# sensor, so this still works on a task that has no thermal_viz configured.
_viz_cam = env.unwrapped.scene.sensors.get("thermal_viz")
_panel_cam = _viz_cam if _viz_cam is not None else thermal_cam
if _panel_cam is not None:
    _THERMAL_PANEL_WH = (_panel_cam.cfg.width, _panel_cam.cfg.height)
    print(f"[INFO] thermal panel drawn from {_THERMAL_PANEL_WH[0]}x{_THERMAL_PANEL_WH[1]} camera")

lidar = env.unwrapped.scene.sensors.get("lidar")
if lidar is not None:
    from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

    print("[INFO] LiDAR present -- rendering bird's-eye scope panel")

_PANEL = 360  # side of each right-hand panel; chase is 1280x720, so 2 stack exactly
_LIDAR_MAX_RANGE = 10.0  # must match MultiMeshRayCasterCfg.max_distance


def _ironbow_lut() -> np.ndarray:
    """256-entry black -> purple -> red -> orange -> yellow -> white ramp.

    Grayscale is technically honest but reads as a black square on screen, which made the
    thermal panel look broken rather than cold. Real thermal cameras ship an ironbow palette
    for exactly this reason. Built with np.interp so it costs nothing and adds no dependency
    -- matplotlib for one gradient would be absurd.
    """
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = [0, 60, 200, 255, 255]
    g = [0, 0, 40, 170, 255]
    b = [0, 90, 40, 20, 220]
    x = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(x, stops, c) for c in (r, g, b)], axis=1).astype(np.uint8)


_LUT = _ironbow_lut()


def _label(img, text, x, y):
    """Small caption used to mark each panel, so the split screen explains itself."""
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    left, top, right, bottom = d.textbbox((x, y), text, font=font)
    d.rectangle([left - 8, top - 6, right + 8, bottom + 6], fill=(0, 0, 0))
    d.text((x, y), text, fill=(235, 235, 235), font=font)


def _lidar_panel():
    """Bird's-eye scope of the 288-ray LiDAR, robot at centre, forward pointing up.

    A 360-degree sensor has no meaningful "POV image", so a top-down scope is the honest
    rendering -- it is also what an operator console would actually show. Hits are rotated
    out of world frame into the robot's own frame, so the plot stays stable while the robot
    turns instead of spinning with it.

    Deliberately not built on rescue_mdp.lidar_ranges: that returns normalized scalar ranges
    for the policy and throws away the bearing, which is the only thing a scope needs.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_PANEL, _PANEL), (8, 12, 16))
    d = ImageDraw.Draw(img)
    c = _PANEL // 2
    scale = (_PANEL / 2 - 15) / _LIDAR_MAX_RANGE  # px per metre

    for ring_m in (2.0, 5.0, 10.0):
        r = ring_m * scale
        d.ellipse([c - r, c - r, c + r, c + r], outline=(38, 52, 62))

    hits_w = lidar.data.ray_hits_w.torch[0]  # (rays, 3)
    pos_w = lidar.data.pos_w.torch[0]  # (3,)
    quat_w = lidar.data.quat_w.torch[0]  # (4,)
    rel = hits_w - pos_w.unsqueeze(0)
    local = quat_apply_inverse(quat_w.unsqueeze(0).expand(rel.shape[0], 4), rel)

    # Misses come back as inf hit positions; a saturated ray carries no information about
    # the world, so drawing it at max range would paint a fake wall around the robot.
    finite = torch.isfinite(local).all(dim=-1)
    pts = local[finite].cpu().numpy()
    xy, z = pts[:, :2], pts[:, 2]
    rng = np.linalg.norm(xy, axis=1)
    keep = rng < _LIDAR_MAX_RANGE * 0.99
    xy, z = xy[keep], z[keep]

    # Colour by height, not by range. The fan is tilted down (-25 to +5 deg), so on open
    # ground most rays come back as a bland ring of floor hits and a range-coloured scope
    # is mostly decoration. Height separates what the robot has to walk around -- steps,
    # rubble, casualties -- from the floor it walks on. Sensor sits ~0.75 m up, so ground
    # returns land near z = -0.75 and anything raised climbs toward 0.
    for (fx, fy), pz in zip(xy, z):
        # Robot frame is x forward, y left. Screen is y down, x right -- so forward maps to
        # -screen_y and left maps to -screen_x.
        px = c - fy * scale
        py = c - fx * scale
        t = float(np.clip((pz + 0.80) / 0.80, 0.0, 1.0))  # 0 = floor, 1 = raised
        colour = (int(40 + 215 * t), int(120 + 135 * t), int(110 * (1.0 - t) + 40))
        r_px = 1.4 + 1.2 * t  # raised hits drawn larger so they read at a glance
        d.ellipse([px - r_px, py - r_px, px + r_px, py + r_px], fill=colour)

    d.polygon([(c, c - 9), (c - 6, c + 7), (c + 6, c + 7)], fill=(235, 235, 235))
    return img


# Radiometry for the *displayed* image only. See _apply_ir_realism.
_IR_ATTENUATION_M = 12.0
_IR_NETD_C = 0.35
_IR_BLUR_PX = 0.6


def _apply_ir_realism(temp_c, dist_m):
    """Make a class-LUT temperature image behave like an IR sensor reading.

    ``thermal.py`` paints every casualty pixel a flat 36.5 C no matter how far away it is,
    because it is a semantic-class lookup, not a radiometric model. That is fine as a policy
    observation -- the network only needs a consistent signal -- but on screen it looks like
    a diagram rather than a sensor, which is what prompted this.

    Three effects, all cheap and all in the right direction:

    * **Atmospheric attenuation.** Radiance falls off toward ambient with path length
      (Beer-Lambert), so a distant casualty reads cooler than a near one. At the 12 m
      attenuation length used here a body at 6.5 m retains ~58% of its contrast over
      ambient -- visibly dimmer, still clearly there, which is the honest depiction of a
      casualty near the edge of sensor range.
    * **NETD.** Real microbolometers quote a noise-equivalent temperature difference,
      typically 30-60 mK for good hardware. 0.35 C is deliberately worse than a real
      camera: at 64x64 upscaled to 360 px, a physically accurate 0.05 C would be invisible,
      and the point is to show that the sensor is noisy rather than to certify a datasheet.
    * **Optical blur.** No lens resolves a step edge, so the sharp segmentation boundary is
      softened slightly.

    IMPORTANT: this runs on the viz camera only and never on the observation path. The
    policy's thermal input must stay byte-identical to what it trained on, so
    rescue_mdp.thermal_image and thermal.py are deliberately left alone. Do not be tempted
    to "fix" the flat image at the source -- that invalidates every trained checkpoint.
    """
    ambient = _thermal_utils.AMBIENT_TEMPERATURE_C
    # Attenuate contrast over ambient, not absolute temperature: an empty scene must stay at
    # ambient rather than decaying toward absolute zero.
    contrast = temp_c - ambient
    if dist_m is not None:
        # Reshape the depth AOV to match the temperature image rather than trusting its
        # layout. distance_to_image_plane carries a trailing channel axis in some Isaac Lab
        # versions and not others, and silently broadcasting (N,H,W) against (N,H,W,1)
        # produces a (N,H,W,W) tensor that survives every arithmetic step and only explodes
        # later, in PIL, as an unreadable "cannot handle this data type" -- which is exactly
        # how this was found. Compare element counts and reshape; never rely on broadcast.
        d = torch.nan_to_num(dist_m.to(temp_c.dtype), nan=1e3, posinf=1e3)
        if d.shape != temp_c.shape:
            if d.numel() == temp_c.numel():
                d = d.reshape(temp_c.shape)
            else:
                print(
                    f"[WARN] depth shape {tuple(d.shape)} incompatible with thermal "
                    f"{tuple(temp_c.shape)} -- skipping distance attenuation"
                )
                d = None
        if d is not None:
            contrast = contrast * torch.exp(-d.clamp(min=0.0) / _IR_ATTENUATION_M)
    out = ambient + contrast
    if _IR_NETD_C > 0:
        out = out + torch.randn_like(out) * _IR_NETD_C
    return out


def _thermal_panel():
    """False-colour pseudo-thermal image, upscaled to the panel size."""
    from PIL import Image, ImageFilter

    seg = _panel_cam.data.output["semantic_segmentation"].torch
    temp = _thermal_utils.segmentation_to_temperature(seg)
    # Best-effort: the radiometry is decoration, the panel is the deliverable. An earlier
    # version let a shape bug inside it abort the whole render, and twelve consecutive takes
    # died for a cosmetic effect. Never again -- fall back to the flat image instead.
    # Only dress up the VIZ camera. When the panel is the policy's own buffer the caption says
    # (POLICY), and adding NETD noise on top would make that caption a lie -- the actor
    # receives the flat class-LUT image, noise and all absent. Caught by looking at a frame
    # and reading the label against the pixels.
    if _panel_cam is not thermal_cam:
        try:
            _depth = _panel_cam.data.output.get("distance_to_image_plane")
            temp = _apply_ir_realism(temp, _depth.torch if _depth is not None else None)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; nothing here is fatal
            print(f"[WARN] IR realism failed ({exc}) -- rendering flat thermal image")
            temp = _thermal_utils.segmentation_to_temperature(seg)
    norm = _thermal_utils.normalize_temperature(temp)[0]
    gray = (norm.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    # The LUT indexes to (..., 3), so gray must be exactly 2-D or PIL gets an array it cannot
    # read. Assert the shape here, where the cause is visible, rather than letting it surface
    # as a type error three calls away.
    if gray.ndim != 2:
        raise RuntimeError(
            f"thermal panel expected a 2-D image, got {gray.shape} -- "
            f"camera is {_THERMAL_PANEL_WH[0]}x{_THERMAL_PANEL_WH[1]}"
        )
    # Interpolation follows the source resolution, and the reason is honesty rather than
    # taste. NEAREST on a 16x16 buffer is deliberate: smoothing it would show the viewer
    # detail the policy does not receive. At 64x64 that argument is gone -- the blocks are
    # now an artefact of a 5.6x upscale rather than a property of the sensor -- so BILINEAR
    # is the more faithful choice.
    resample = Image.NEAREST if max(_THERMAL_PANEL_WH) <= 16 else Image.BILINEAR
    img = Image.fromarray(_LUT[gray]).resize((_PANEL, _PANEL), resample)
    # Optical blur, applied after the upscale so the radius is in display pixels. No lens
    # resolves a step edge, and the segmentation buffer is all step edges. Skipped at 16x16,
    # where NEAREST blocks are the honest depiction and softening them would misrepresent
    # what the policy receives.
    if _IR_BLUR_PX > 0 and max(_THERMAL_PANEL_WH) > 16:
        img = img.filter(ImageFilter.GaussianBlur(_IR_BLUR_PX * _PANEL / _THERMAL_PANEL_WH[0]))
    return img


def _thermal_label() -> str:
    """Panel caption, derived from the real camera rather than hardcoded.

    Says VIZ whenever the panel is not the policy's own buffer, so the video can never be
    read as a claim that the actor sees this resolution -- which is the exact claim the
    THERMAL_RESOLUTION measurement disproves.
    """
    w, h = _THERMAL_PANEL_WH
    tag = "VIZ" if _panel_cam is not thermal_cam else "POLICY"
    return f"THERMAL {w}x{h} ({tag})"


def _composite(rgb_frame: np.ndarray) -> np.ndarray:
    """Chase view left; thermal POV and LiDAR scope stacked in a right-hand column.

    Three views because the robot has three ways of perceiving and a demo that shows only
    the outside view proves nothing about the sensors. Panel size is chosen so two of them
    stack to exactly the chase camera's 720 px height -- no letterbox bars, no stretching.
    """
    panels = []
    if thermal_cam is not None and _thermal_utils is not None:
        panels.append((_thermal_label(), _thermal_panel()))
    if lidar is not None:
        panels.append(("LIDAR 288 rays", _lidar_panel()))
    if not panels:
        return rgb_frame

    h, w = rgb_frame.shape[0], rgb_frame.shape[1]
    out = np.zeros((h, w + _PANEL, 3), dtype=np.uint8)
    out[:, :w] = rgb_frame
    for i, (_, panel) in enumerate(panels):
        y = i * _PANEL
        if y + _PANEL <= h:
            out[y : y + _PANEL, w:] = np.asarray(panel)
    return out


def _panel_labels(img, chase_w):
    """Label each right-column panel in place, matching _composite's stacking order."""
    names = []
    if thermal_cam is not None and _thermal_utils is not None:
        names.append(_thermal_label())
    if lidar is not None:
        names.append("LIDAR 288 rays")
    for i, name in enumerate(names):
        _label(img, name, chase_w + 12, i * _PANEL + 12)

# Voice communication (plan Phase 4, item 14). Explicitly NOT part of the RL loop: the
# policy is never rewarded for talking. This is the radio interface bolted on top -- on
# reaching a casualty the robot emits a scripted call, logged as JSONL and burned into the
# frame so the demo video shows it happening.
#
# The trigger reuses the reward manager's own victim_found term rather than re-deriving
# "found" here. Calling the configured term with its configured params means the caption
# cannot drift from the Episode_Reward/victim_found metric it is supposed to illustrate.
_VOICE_LINE = "CASUALTY LOCATED - relaying position to base"
_found_term = None
try:
    _found_term = env.unwrapped.reward_manager.get_term_cfg("victim_found")
except (AttributeError, ValueError):
    print("[WARN] no victim_found reward term -- voice comms disabled for this task")

_hold_frames = int(round(2.0 / env.unwrapped.step_dt))  # caption dwells ~2 s
_hold = 0
_was_found = False
_voice_log = open(f"{args.out}/voice_log.jsonl", "w")


def _caption(img):
    """Burn the radio call into the frame. Bottom-left, on a dark band for legibility."""
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()  # bitmap fallback; tiny but always present
    draw = ImageDraw.Draw(img)
    x, y = 24, img.height - 60
    text = f"[VOICE] {_VOICE_LINE}"
    # Measure the text instead of guessing a box width. A fixed width was wrong the moment
    # the font or the message changed -- the first render ran the tail of the line off the
    # end of its own background band.
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - 12, top - 10, right + 12, bottom + 10], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 96, 64), font=font)
    return img


# Episode outcome, so the chain driving this can retry a bad take instead of shipping a
# demo of the robot falling over 3 s in. Written to summary.json at the end.
_ep_found = False
_ep_fell = False
_term_mgr = env.unwrapped.termination_manager
_has_fall_term = "base_contact" in _term_mgr.active_terms

saved = 0
for step in range(args.steps):
    with torch.inference_mode():
        # Refresh the navigator's delta at its own rate. The command term keeps applying the
        # last delta on intervening steps, matching how training held it for inner_steps.
        if _nav_policy is not None and step % args.inner_steps == 0:
            _nav_cmd.set_delta(_nav_policy(obs))
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

    if _has_fall_term and bool(dones[0]) and bool(_term_mgr.get_term("base_contact")[0]):
        _ep_fell = True

    if _found_term is not None:
        found = bool(_found_term.func(env.unwrapped, **_found_term.params)[0] > 0.0)
        # Rising edge only: victim_found stays 1.0 for every step the robot loiters inside
        # found_radius, and a radio that repeats itself 50 times a second is noise.
        if found and not _was_found:
            _voice_log.write(
                json.dumps({"step": step, "t": round(step * env.unwrapped.step_dt, 3), "msg": _VOICE_LINE})
                + "\n"
            )
            _voice_log.flush()
            print(f"[VOICE] t={step * env.unwrapped.step_dt:.2f}s {_VOICE_LINE}")
            _hold = _hold_frames
            _ep_found = True
        _was_found = found

    rgb = cam.data.output["rgb"].torch  # (N, H, W, 3 or 4) uint8
    chase_w = rgb.shape[2]
    frame = _composite(rgb[0, ..., :3].cpu().numpy().astype(np.uint8))
    try:
        from PIL import Image

        img = Image.fromarray(frame)
        _label(img, "CHASE CAM", 24, 24)
        _panel_labels(img, chase_w)
        if _hold > 0:
            img = _caption(img)
            _hold -= 1
        img.save(f"{args.out}/f{saved:05d}.png")
    except ImportError:
        np.save(f"{args.out}/f{saved:05d}.npy", frame)
    saved += 1

_voice_log.close()

with open(f"{args.out}/summary.json", "w") as f:
    json.dump(
        {
            "task": args.task,
            "checkpoint": args.checkpoint,
            # Null here means the scripted seeker drove this take, not the navigator. Recorded
            # because the video cannot show the difference: on the Nav task an absent
            # --navigator_checkpoint leaves the residual delta at zeros, which films as a
            # perfectly good demo of the *scripted* controller. Without this field the mp4 has
            # no provenance and a caption claiming a trained policy cannot be checked.
            "navigator_checkpoint": args.navigator_checkpoint,
            # Which scene this was filmed in. A no-rubble take looks like a much more
            # capable robot than the numbers describe, so the video must carry the flag.
            "no_debris": bool(args.no_debris),
            "inner_steps": args.inner_steps if args.navigator_checkpoint else None,
            "seed": args.seed,
            "steps": saved,
            "found_victim": _ep_found,
            "fell": _ep_fell,
        },
        f,
        indent=2,
    )

print(f"[INFO] wrote {saved} frames to {args.out} | found={_ep_found} fell={_ep_fell}")
env.close()
simulation_app.close()

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
    d.rectangle([x - 8, y - 6, x + 9 * len(text) + 8, y + 26], fill=(0, 0, 0))
    d.text((x, y), text, fill=(235, 235, 235), font=font)


def _composite(rgb_frame: np.ndarray) -> np.ndarray:
    """Chase view on the left, upscaled false-colour thermal view on the right."""
    if thermal_cam is None or _thermal_utils is None:
        return rgb_frame
    from PIL import Image

    seg = thermal_cam.data.output["semantic_segmentation"].torch
    norm = _thermal_utils.normalize_temperature(
        _thermal_utils.segmentation_to_temperature(seg)
    )[0]
    gray = (norm.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    h = rgb_frame.shape[0]
    # NEAREST on purpose: the thermal sensor really is 16x16, and smoothing it would
    # misrepresent the resolution the policy actually receives.
    panel = Image.fromarray(_LUT[gray]).resize((h, h), Image.NEAREST)
    out = np.zeros((h, rgb_frame.shape[1] + h, 3), dtype=np.uint8)
    out[:, : rgb_frame.shape[1]] = rgb_frame
    out[:, rgb_frame.shape[1] :] = np.asarray(panel)
    return out

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
    draw.rectangle([x - 12, y - 10, x + 620, y + 38], fill=(0, 0, 0))
    draw.text((x, y), f"[VOICE] {_VOICE_LINE}", fill=(255, 96, 64), font=font)
    return img


saved = 0
for step in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

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
        _was_found = found

    rgb = cam.data.output["rgb"].torch  # (N, H, W, 3 or 4) uint8
    frame = _composite(rgb[0, ..., :3].cpu().numpy().astype(np.uint8))
    try:
        from PIL import Image

        img = Image.fromarray(frame)
        if thermal_cam is not None:
            _label(img, "CHASE CAM", 24, 24)
            _label(img, "THERMAL 16x16", frame.shape[1] - img.height + 24, 24)
        if _hold > 0:
            img = _caption(img)
            _hold -= 1
        img.save(f"{args.out}/f{saved:05d}.png")
    except ImportError:
        np.save(f"{args.out}/f{saved:05d}.npy", frame)
    saved += 1

_voice_log.close()

print(f"[INFO] wrote {saved} frames to {args.out}")
env.close()
simulation_app.close()

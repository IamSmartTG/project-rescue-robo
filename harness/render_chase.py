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
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

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
)

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


def _composite(rgb_frame: np.ndarray) -> np.ndarray:
    """Chase view on the left, upscaled thermal view on the right."""
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
    panel = Image.fromarray(gray).convert("RGB").resize((h, h), Image.NEAREST)
    out = np.zeros((h, rgb_frame.shape[1] + h, 3), dtype=np.uint8)
    out[:, : rgb_frame.shape[1]] = rgb_frame
    out[:, rgb_frame.shape[1] :] = np.asarray(panel)
    return out

saved = 0
for step in range(args.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

    rgb = cam.data.output["rgb"].torch  # (N, H, W, 3 or 4) uint8
    frame = _composite(rgb[0, ..., :3].cpu().numpy().astype(np.uint8))
    try:
        from PIL import Image

        Image.fromarray(frame).save(f"{args.out}/f{saved:05d}.png")
    except ImportError:
        np.save(f"{args.out}/f{saved:05d}.npy", frame)
    saved += 1

print(f"[INFO] wrote {saved} frames to {args.out}")
env.close()
simulation_app.close()

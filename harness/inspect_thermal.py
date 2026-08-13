"""Dump thermal frames and victim geometry so a human can check the sensor actually works.

Everything so far proves the pipeline *runs*: exit 0, correct buffer shape, rewards firing.
None of it proves victims read hot and terrain reads cold, or that the annulus sampling
really scatters the casualties. This renders a few frames to PNG and prints the victim
radii/bearings so both are verifiable by eye.

Usage (on a box where Vulkan works -- not RunPod):
    ./isaaclab.sh -p harness/inspect_thermal.py --num_envs 4 --steps 40 --out ~/thermal_out
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--out", type=str, default="thermal_out")
parser.add_argument("--task", type=str, default="Isaac-Rescue-Anymal-C-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # mandatory: without it the RTX renderer never starts

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Everything below must be imported AFTER the app exists -- importing Isaac Lab runtime
# classes before SimulationApp is what caused the exit-139 SIGSEGV earlier in this project.
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers the gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg)
obs, _ = env.reset()

scene = env.unwrapped.scene
actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
for _ in range(args.steps):
    obs, *_ = env.step(actions)

# --- victim geometry: does the annulus sampling actually scatter them? ---
origins = scene.env_origins  # (N, 3)
print("\n=== victim placement (should differ per env, radius in [1.6, 3.6]) ===")
for name in ("victim_0", "victim_1", "victim_2"):
    pos = scene[name].data.root_pos_w.torch  # (N, 3)
    rel = pos[:, :2] - origins[:, :2]
    radius = rel.norm(dim=-1)
    bearing = torch.rad2deg(torch.atan2(rel[:, 1], rel[:, 0]))
    print(f"{name}: radius={[round(float(r), 2) for r in radius]}  "
          f"bearing_deg={[round(float(b)) for b in bearing]}")

# --- thermal frames ---
cam = scene.sensors["thermal_camera"]
seg = cam.data.output["semantic_segmentation"].torch  # (N, H, W, 4) uint8
print(f"\n=== segmentation buffer: shape={tuple(seg.shape)} dtype={seg.dtype} ===")
print("unique colours (a victim class should be distinct from ground/robot):")
flat = seg.reshape(-1, seg.shape[-1])
uniq, counts = torch.unique(flat, dim=0, return_counts=True)
for c, n in zip(uniq.tolist(), counts.tolist()):
    print(f"  rgba={c}  pixels={n}")

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c_rescue import (  # noqa: E402
    thermal as thermal_utils,
)

temperature = thermal_utils.segmentation_to_temperature(seg)      # (N, H, W)
normalized = thermal_utils.normalize_temperature(temperature)     # (N, H, W) in [0, 1]
print(f"\n=== temperature: min={float(normalized.min()):.3f} "
      f"max={float(normalized.max()):.3f} mean={float(normalized.mean()):.3f} ===")
if float(normalized.max()) - float(normalized.min()) < 1e-6:
    print("WARNING: thermal image is uniform -- nothing is registering as warm. "
          "Check semantic tags and the SEMANTIC_FILTER.")

try:
    from PIL import Image

    for i in range(min(args.num_envs, 4)):
        Image.fromarray(seg[i, ..., :3].cpu().numpy().astype(np.uint8)).save(
            f"{args.out}/env{i}_segmentation.png"
        )
        gray = (normalized[i].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(gray).save(f"{args.out}/env{i}_thermal.png")
    print(f"\nwrote PNGs to {args.out}")
except ImportError:
    np.save(f"{args.out}/thermal.npy", normalized.cpu().numpy())
    print(f"\nPillow unavailable; wrote raw array to {args.out}/thermal.npy")

env.close()
simulation_app.close()

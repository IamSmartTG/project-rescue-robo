"""Evaluate a trained rescue policy and report the numbers the brief actually asks for.

Why this exists: every figure in the logs so far is a *training* metric averaged over a
running window while the policy is still changing and the terrain curriculum is still
moving. "Episode_Reward/victim_found = 0.14" is not a success rate and must not be quoted
as one. This runs the frozen policy over one full episode per environment and counts what
happened.

Reported:
    success rate     -- fraction of episodes where the robot got within found_radius of a
                        casualty at least once before the episode ended
    time to find     -- seconds to first contact, over successful episodes only
    fall rate        -- fraction of episodes ended by base contact rather than time-out
    terrain level    -- mean difficulty tier the episodes ran on

Usage:
    ./isaaclab.sh -p harness/eval_rescue.py --task Isaac-Rescue-Anymal-C-v0 \
        --checkpoint <path>/model_3199.pt --num_envs 256 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Rescue-Anymal-C-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--out", type=str, default="eval_rescue.json")
# Disables the random shoves only, leaving terrain, payload and observation noise identical.
# Switching to the PLAY task would disable them too but also shrink the terrain grid, which
# would confound the one variable this is meant to isolate.
parser.add_argument("--no_disturbance", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # the thermal sensor is part of the observation the policy expects

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Same ordering constraint as render_chase.py: Isaac Lab runtime classes cannot be imported
# before SimulationApp exists.
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

# Note on terrain difficulty: a fresh env does NOT inherit the 3.66 mean level the training
# run climbed to. TerrainImporter seeds each env's level randomly up to
# ``max_init_terrain_level``, and the curriculum then moves it. So the level reported below
# describes the tiers this eval actually ran on -- it is not a restatement of the training
# curriculum, and must not be read as one.

if args.no_disturbance:
    # Isolates the question "can this robot stay upright on this terrain" from "is this robot
    # being shoved". Both events are active in every eval number recorded so far, so a fall
    # rate measured with them on cannot distinguish the two.
    for term in ("push_robot", "base_external_force_torque"):
        if getattr(env_cfg.events, term, None) is not None:
            setattr(env_cfg.events, term, None)
    print("[INFO] disturbance events disabled (push_robot, base_external_force_torque)")

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
from importlib import metadata  # noqa: E402

agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[INFO] loaded checkpoint: {args.checkpoint}")

dev = env.unwrapped.device
n = args.num_envs

# Reuse the reward manager's own victim_found term as the success test, so this number and
# Episode_Reward/victim_found can never describe different events.
found_term = env.unwrapped.reward_manager.get_term_cfg("victim_found")

found = torch.zeros(n, dtype=torch.bool, device=dev)
find_step = torch.full((n,), -1, dtype=torch.long, device=dev)
done_once = torch.zeros(n, dtype=torch.bool, device=dev)  # freeze stats after first episode
ep_len = torch.zeros(n, dtype=torch.long, device=dev)
fell = torch.zeros(n, dtype=torch.bool, device=dev)

# TerminationManager.get_term returns the flags for the *current* step only -- they are
# overwritten every step. Reading it after the loop would report whatever the last step
# happened to look like, so the fall flag has to be latched as each episode ends.
term = env.unwrapped.termination_manager
has_fall_term = "base_contact" in term.active_terms

# One full episode is max_episode_length steps; add margin so time-outs are captured too.
horizon = int(env.unwrapped.max_episode_length) + 10
step_dt = env.unwrapped.step_dt

obs = env.get_observations()
for step in range(horizon):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

    live = ~done_once
    if not live.any():
        break

    hit = found_term.func(env.unwrapped, **found_term.params) > 0.0
    first = hit & live & ~found
    find_step = torch.where(first, torch.full_like(find_step, step), find_step)
    found |= hit & live
    ep_len = torch.where(live, ep_len + 1, ep_len)

    # Latch the ending condition for envs finishing their first episode on this step.
    ending = dones.bool() & live
    if has_fall_term and ending.any():
        fell |= ending & term.get_term("base_contact").bool()

    done_once |= dones.bool()

print(f"[INFO] {int(done_once.sum())}/{n} episodes terminated within {horizon} steps")

succ = found.float().mean().item()
ttf = find_step[found]
result = {
    "checkpoint": args.checkpoint,
    "task": args.task,
    "episodes": n,
    "success_rate": round(succ, 4),
    "mean_time_to_find_s": round((ttf.float().mean().item() * step_dt), 3) if ttf.numel() else None,
    "mean_episode_length_steps": round(ep_len.float().mean().item(), 1),
    "mean_episode_length_s": round(ep_len.float().mean().item() * step_dt, 2),
    "fall_rate": round(fell.float().mean().item(), 4),
    "mean_terrain_level": round(env.unwrapped.scene.terrain.terrain_levels.float().mean().item(), 3)
    if getattr(env.unwrapped.scene, "terrain", None) is not None
    and getattr(env.unwrapped.scene.terrain, "terrain_levels", None) is not None
    else None,
}

print(json.dumps(result, indent=2))
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(f"[INFO] wrote {args.out}")

env.close()
simulation_app.close()

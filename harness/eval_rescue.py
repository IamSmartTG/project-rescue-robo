"""Evaluate a trained rescue policy and report the numbers the brief actually asks for.

Why this exists: every figure in the logs so far is a *training* metric averaged over a
running window while the policy is still changing and the terrain curriculum is still
moving. "Episode_Reward/victim_found = 0.14" is not a success rate and must not be quoted
as one. This runs the frozen policy over one full episode per environment and counts what
happened.

Reported:
    success rate     -- fraction of episodes where the robot got within found_radius of a
                        casualty at least once before the episode ended
    SPL              -- success weighted by path length, the standard target-finding metric
                        (see :mod:`spl`). Success rate alone cannot separate "searched and
                        found" from "wandered into one"; this can.
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
# The emergency-supply payload is carried in every measurement taken so far. 2-12 kg plus
# randomized inertia on a ~30 kg trunk is a large handicap and an obvious fall suspect.
parser.add_argument("--no_payload", action="store_true")
# Without this, evaluating the Nav task measures the SCRIPTED SEEKER, not a trained
# navigator: nothing calls set_delta(), so the residual stays at its initialised zeros. That
# is exactly how the scripted baseline was produced -- and exactly how a "trained navigator"
# result could silently be the baseline wearing a different label.
parser.add_argument("--navigator_checkpoint", type=str, default=None)
parser.add_argument("--inner_steps", type=int, default=5)
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
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

# Sibling module in harness/, resolved through sys.path[0]. Pure Python and torch-free on
# purpose -- the SPL reduction runs once over a few hundred finished episodes, so it costs
# nothing to keep it testable on a laptop with no Isaac Sim and no torch.
import spl as spl_utils  # noqa: E402

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

if args.no_payload:
    # Note this measures an unloaded robot the policy was never trained as -- it trained
    # always carrying something. A drop in fall rate here means the payload is the handicap;
    # a rise would mean the policy has come to rely on the added mass.
    for term in ("add_payload_mass", "add_payload_inertia"):
        if getattr(env_cfg.events, term, None) is not None:
            setattr(env_cfg.events, term, None)
    print("[INFO] emergency-supply payload disabled")

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
from importlib import metadata  # noqa: E402

agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[INFO] loaded checkpoint: {args.checkpoint}")

_nav_policy = None
_nav_cmd = None
if args.navigator_checkpoint:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c_rescue.nav_env import (  # noqa: E402
        NavigatorEnv,
    )

    _nav_cmd = env.unwrapped.command_manager.get_term("base_velocity")
    if not hasattr(_nav_cmd, "set_delta"):
        raise SystemExit(
            f"--navigator_checkpoint given but base_velocity is {type(_nav_cmd).__name__}; "
            "use the -Nav- task."
        )
    _nav_shim = NavigatorEnv(env, policy, _nav_cmd, args.inner_steps)
    _nav_cfg = load_cfg_from_registry(args.task, "navigator_cfg_entry_point")
    _nav_cfg = handle_deprecated_rsl_rl_cfg(_nav_cfg, metadata.version("rsl-rl-lib"))
    _nav_runner = OnPolicyRunner(_nav_shim, _nav_cfg.to_dict(), log_dir=None, device=args.device)
    _nav_runner.load(args.navigator_checkpoint)
    _nav_policy = _nav_runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] navigator ACTIVE: {args.navigator_checkpoint}")
else:
    print("[INFO] no navigator -- measuring the scripted seeker (residual delta stays zero)")

dev = env.unwrapped.device
n = args.num_envs

# Reuse the reward manager's own victim_found term as the success test, so this number and
# Episode_Reward/victim_found can never describe different events.
found_term = env.unwrapped.reward_manager.get_term_cfg("victim_found")

# Read the search geometry off the reward term rather than restating it, so SPL can never
# describe a different task than success_rate does. Changing found_radius or the victim roster
# in rescue_env_cfg.py then flows through here with no edit.
victim_names = found_term.params["victim_names"]
found_radius = found_term.params["found_radius"]
print(f"[INFO] {len(victim_names)} victim asset(s): {victim_names}, found_radius={found_radius} m")

robot = env.unwrapped.scene["robot"]

# Rubble props, discovered from the scene rather than imported from the config, so this keeps
# working on tasks that have none (the pre-debris checkpoints, the NoCam variant) instead of
# failing on a missing name.
debris_names = tuple(
    sorted(k for k in getattr(env.unwrapped.scene, "rigid_objects", {}) if k.startswith("debris"))
)
# Termination within this distance of a rubble centre is counted as "hit the rubble".
# The block is 0.8 m across and the robot is ~1.0 m long, so contact happens with centres
# roughly 0.9 m apart; 1.5 m leaves margin for the trunk's own length and for the block
# having been nudged. Too tight and real collisions get filed as terrain falls.
debris_near_m = 1.5
if debris_names:
    print(f"[INFO] {len(debris_names)} debris prop(s): {debris_names}")
else:
    print("[INFO] no debris props in this task -- collision split not reported")

found = torch.zeros(n, dtype=torch.bool, device=dev)
fell_near_debris = torch.zeros(n, dtype=torch.bool, device=dev)
# Fraction of steps on which the LiDAR shield vetoed forward motion. This distinguishes a
# navigator that learned to avoid debris from one that merely gets rescued by the shield on
# every approach -- without it, "learned obstacle avoidance" is an unfalsifiable claim.
shield_steps = torch.zeros(n, device=dev)
_shield_cmd = env.unwrapped.command_manager.get_term("base_velocity")
_has_shield = hasattr(_shield_cmd, "shield_fired")
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

# SPL needs the shortest path that would have counted as success, measured at spawn, and the
# path actually walked to first contact. Reuse the reward code's own distance helper rather
# than recomputing "distance to nearest victim" a second way -- two implementations of that
# would eventually disagree.
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c_rescue.rescue_mdp import (  # noqa: E402
    _nearest_victim,
)

# Captured before the first step(), so these are spawn positions.
spawn_dist, _ = _nearest_victim(env.unwrapped, victim_names, SceneEntityCfg("robot"))
spawn_dist = spawn_dist.clone()
path_len = torch.zeros(n, device=dev)
prev_pos = robot.data.root_pos_w.torch[:, :2].clone()

for step in range(horizon):
    with torch.inference_mode():
        if _nav_policy is not None and step % args.inner_steps == 0:
            _nav_cmd.set_delta(_nav_policy(obs))
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

    live = ~done_once
    if not live.any():
        break

    # Path walked, for SPL. Three gates, each load-bearing:
    #   live      -- stop counting once this env's first episode is over
    #   ~found    -- SPL measures the path *to* the goal, not wandering afterwards
    #   ~dones    -- ManagerBasedRLEnv resets terminated envs *inside* step(), so on the
    #                terminating step root_pos_w has already teleported to the next spawn.
    #                Without this the delta adds metres of fictional path. It only ever
    #                mattered for failed episodes (a found env is already excluded), and
    #                those contribute 0 to SPL -- but they do contribute to
    #                mean_path_length_m, which would otherwise be nonsense.
    if _has_shield:
        shield_steps += torch.where(live & _shield_cmd.shield_fired, 1.0, 0.0)

    cur_pos = robot.data.root_pos_w.torch[:, :2]
    moving = live & ~found & ~dones.bool()
    path_len += torch.where(moving, torch.norm(cur_pos - prev_pos, dim=-1), torch.zeros_like(path_len))
    prev_pos = cur_pos.clone()

    hit = found_term.func(env.unwrapped, **found_term.params) > 0.0
    first = hit & live & ~found
    find_step = torch.where(first, torch.full_like(find_step, step), find_step)
    found |= hit & live
    ep_len = torch.where(live, ep_len + 1, ep_len)

    # Latch the ending condition for envs finishing their first episode on this step.
    ending = dones.bool() & live
    if has_fall_term and ending.any():
        this_fall = ending & term.get_term("base_contact").bool()
        fell |= this_fall
        # Split falls into "hit the rubble" and "lost it on the terrain".
        #
        # Without this the headline fall rate is uninterpretable now that debris exists:
        # base_contact fires on trunk contact with anything, so a run that goes 0.39 -> 0.60
        # could mean the obstacles are real and the robot is bumping them (expected, and the
        # thing being measured) or that the walker has degraded (a regression). Those demand
        # opposite responses and the old metric cannot tell them apart.
        #
        # Proximity heuristic, not ground truth: ContactSensor reports net force on the
        # trunk, not which body applied it, so identifying the collision partner properly
        # would mean a per-pair contact filter. Terminating within debris_near_m of a block
        # is a good enough proxy -- and it is a proxy, so the field is named for the
        # measurement rather than for the conclusion.
        if debris_names:
            d_debris, _ = _nearest_victim(env.unwrapped, debris_names, SceneEntityCfg("robot"))
            fell_near_debris |= this_fall & (d_debris < debris_near_m)

    done_once |= dones.bool()

print(f"[INFO] {int(done_once.sum())}/{n} episodes terminated within {horizon} steps")

succ = found.float().mean().item()
ttf = find_step[found]

# SPL. The reduction is a few hundred scalars, so it drops to plain Python lists rather than
# growing a torch implementation that the self-check in spl.py could not exercise.
shortest = [spl_utils.shortest_path_length(d, found_radius) for d in spawn_dist.tolist()]
spl_score = spl_utils.spl(found.tolist(), path_len.tolist(), shortest)

result = {
    "checkpoint": args.checkpoint,
    "task": args.task,
    "navigator": args.navigator_checkpoint,
    "episodes": n,
    "victims": len(victim_names),
    "success_rate": round(succ, 4),
    # Straight-line, not geodesic -- see the module docstring in spl.py. Named so that the
    # approximation travels with the number instead of being lost on the way into a table.
    "spl_straight_line_proxy": round(spl_score, 4),
    "mean_path_length_m": round(path_len.mean().item(), 2),
    "mean_shortest_path_m": round(sum(shortest) / len(shortest), 2) if shortest else None,
    "mean_time_to_find_s": round((ttf.float().mean().item() * step_dt), 3) if ttf.numel() else None,
    "mean_episode_length_steps": round(ep_len.float().mean().item(), 1),
    "mean_episode_length_s": round(ep_len.float().mean().item() * step_dt, 2),
    "fall_rate": round(fell.float().mean().item(), 4),
    # Split of the line above, not an addition to it. Named "_near_debris" rather than
    # "_by_collision" because it is a proximity proxy: the contact sensor reports net force
    # on the trunk, not what applied it. Reported only when the task has rubble at all.
    "fall_rate_near_debris": round(fell_near_debris.float().mean().item(), 4) if debris_names else None,
    "fall_rate_terrain_only": round((fell & ~fell_near_debris).float().mean().item(), 4)
    if debris_names
    else None,
    "debris_props": len(debris_names),
    # Mean fraction of live steps where the shield vetoed forward motion. High late in
    # training means the policy is leaning on the shield rather than steering around debris.
    "shield_veto_rate": round((shield_steps / ep_len.clamp(min=1).float()).mean().item(), 4)
    if _has_shield
    else None,
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

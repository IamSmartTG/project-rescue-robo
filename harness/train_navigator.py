"""Train the residual navigator on top of the frozen locomotion policy.

The hierarchy, and who runs at what rate::

    navigator (trained here)   ~10 Hz   sees thermal + LiDAR, emits a velocity-command delta
    scripted seeker            50 Hz    base command from the thermal centroid
    LiDAR shield               50 Hz    vetoes forward motion near an obstacle
    walker (model_4197)        50 Hz    frozen; tracks whatever command it is handed

Only the navigator has trainable weights. The walker is loaded, frozen, and treated as part
of the environment -- which is the entire reason this is affordable: locomotion on rough
terrain took ~5200 iterations to learn and is not being relearned.

Decision rate is the one number here worth arguing about. The navigator acts every
``--inner_steps`` physics steps (default 5 -> 10 Hz). Acting at the full 50 Hz would need no
inner loop at all and would be simpler, but with ``gamma=0.99`` the effective horizon is
~100 steps = 2 s, far short of a 20 s episode, so credit for reaching a casualty never
reaches the decision that turned toward it. Decimating by 5 stretches the same discount over
10 s. This is the standard reason hierarchical controllers run their high level slowly, not
an optimisation.

Usage:
    ./isaaclab.sh -p harness/train_navigator.py \
        --task Isaac-Rescue-Anymal-C-Nav-v0 \
        --walker_checkpoint logs/rsl_rl/anymal_c_rescue/<run>/model_4197.pt \
        --num_envs 4096 --max_iterations 600 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Rescue-Anymal-C-Nav-v0")
parser.add_argument("--walker_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=600)
parser.add_argument("--inner_steps", type=int, default=5, help="physics steps per navigator decision")
parser.add_argument("--log_dir", type=str, default="logs/rsl_rl/navigator")
parser.add_argument(
    "--no_zero_init",
    action="store_true",
    help="skip zeroing the actor's output layer, reproducing the pre-fix runs (iter 0 does "
    "NOT equal the scripted seeker). Only useful for reproducing the Aug 15-16 results.",
)
parser.add_argument(
    "--resume_checkpoint",
    type=str,
    default=None,
    help="navigator model_*.pt to continue from. --max_iterations is ADDITIVE on top of the "
    "iteration stored in it, so 225 -> 300 is --max_iterations 75, not 300.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # the navigator's whole input is a rendered thermal image

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c_rescue.nav_env import (  # noqa: E402
    NavigatorEnv,
)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg)

# The walker's agent cfg, so its network is rebuilt exactly as trained.
walker_cfg = load_cfg_from_registry("Isaac-Rescue-Anymal-C-v0", "rsl_rl_cfg_entry_point")
from importlib import metadata  # noqa: E402

walker_cfg = handle_deprecated_rsl_rl_cfg(walker_cfg, metadata.version("rsl-rl-lib"))
wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(walker_cfg, "clip_actions", None))

walker_runner = OnPolicyRunner(wrapped, walker_cfg.to_dict(), log_dir=None, device=args.device)
walker_runner.load(args.walker_checkpoint)
walker_policy = walker_runner.get_inference_policy(device=env.unwrapped.device)
print(f"[INFO] frozen walker loaded: {args.walker_checkpoint}")

command_term = env.unwrapped.command_manager.get_term("base_velocity")
if not hasattr(command_term, "set_delta"):
    raise SystemExit(
        f"base_velocity is {type(command_term).__name__}, not ResidualNavigatorCommand. "
        "Use the -Nav- task variant; the plain task still has the random UniformVelocityCommand."
    )

nav_env = NavigatorEnv(wrapped, walker_policy, command_term, args.inner_steps)
print(f"[INFO] navigator: obs={nav_env.num_obs} act=3 rate={50 / args.inner_steps:.0f} Hz")

nav_cfg = load_cfg_from_registry(args.task, "navigator_cfg_entry_point")
nav_cfg = handle_deprecated_rsl_rl_cfg(nav_cfg, metadata.version("rsl-rl-lib"))
nav_cfg.max_iterations = args.max_iterations

os.makedirs(args.log_dir, exist_ok=True)
runner = OnPolicyRunner(nav_env, nav_cfg.to_dict(), log_dir=args.log_dir, device=args.device)

def _zero_output_layer(actor) -> str:
    """Zero the actor's last Linear so the residual starts at exactly zero.

    Without this the residual is not a residual. ``init_std=0.3`` in NavigatorPPORunnerCfg
    shrinks the *exploration noise*, but the policy's *mean* action is whatever the untrained
    MLP emits, and evaluation uses the mean. Measured on the Aug 15 run's model_0: a mean
    delta of +0.084 / -0.100 / +0.004 m/s and ~15% of the +/-1.0 command clamp in magnitude,
    including a standing sideways bias. That is not a correction to the scripted seeker, it is
    a corruption of it, and it cost ~12 points of success rate before training began --
    model_0 evaluated at 0.129, statistically identical to the random-command walker's 0.125
    and nowhere near the seeker's 0.25 the design promised.

    Zeroing weight *and* bias makes delta identically zero, so iteration 0 reproduces the
    scripted seeker exactly rather than approximately. Standard practice in residual RL and
    the thing that makes "training can only move away from a known-decent controller" true.
    """
    last = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            last = module  # the final Linear encountered is the output head
    if last is None:
        raise SystemExit("could not find a Linear layer in the actor to zero-init")
    # The head is identified positionally -- the last Linear in modules() order. That is
    # correct for the current actor (MLPModel registers obs_normalizer -> mlp -> distribution
    # and only mlp holds Linears), but it is an assumption, not a guarantee. If a future actor
    # grows a Linear after the mlp this would zero the wrong tensor, the run would look
    # completely normal, and the residual claim would silently be false again. Checking the
    # output width costs nothing and turns that into a startup failure.
    if last.out_features != 3:
        raise SystemExit(
            f"zero-init target has {last.out_features} outputs, expected 3 (vx, vy, wz). "
            "The actor's output head is not where this function assumed it was."
        )
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    return f"{tuple(last.weight.shape)}"


start_it = 0
if args.resume_checkpoint:
    runner.load(args.resume_checkpoint)
    start_it = runner.current_learning_iteration
    # Printed loud because a load that silently starts from 0 produces a full, plausible,
    # worthless run. The first log header must read `start_it/(start_it+max_iterations)`.
    print(f"[INFO] resumed {args.resume_checkpoint} at iteration {start_it}")
    if start_it == 0:
        raise SystemExit(
            f"{args.resume_checkpoint} restored iteration 0 -- that is a fresh start, not a "
            "resume. Check the checkpoint is a navigator model_*.pt, not a walker one."
        )
elif not args.no_zero_init:
    # Fresh start only. Doing this after a resume would zero out trained weights and silently
    # throw the run away, which is a far worse failure than the bug it fixes.
    shape = _zero_output_layer(runner.alg.actor)
    print(f"[INFO] residual zero-init: actor output layer {shape} zeroed -- iter 0 == scripted seeker")
else:
    print("[WARN] --no_zero_init: residual starts NON-zero, iter 0 is NOT the scripted seeker")

print(f"[INFO] training {args.max_iterations} iterations -> {start_it + args.max_iterations}")
runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)

env.close()
simulation_app.close()

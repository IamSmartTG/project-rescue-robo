# Phase 2 Benchmark — ANYmal-C Rough Terrain Baseline

Run date: 2026-08-12
Hardware: RunPod RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM, 256 cores, 1007 GB RAM
Stack: Isaac Sim 6.0.1.0, Isaac Lab (main), RSL-RL PPO, Python 3.12, CUDA 13.0

## Configuration

| Parameter | Value |
|---|---|
| Task | `Isaac-Velocity-Rough-Anymal-C-v0` |
| num_envs | 4096 |
| num_steps_per_env | 24 |
| max_iterations | 1500 |
| Mode | headless (`--viz none`) |

## Throughput

| Metric | Value |
|---|---|
| Steps per iteration | 98,304 (4096 x 24) |
| Iteration time | ~2.4 s |
| **Throughput** | **~41,000 env-steps/sec** |
| Wall clock, 1500 iters | ~53 min (ETA reported at iter 33) |

## Resource utilization (at 4096 envs)

| Resource | Used | Available | Headroom |
|---|---|---|---|
| VRAM | 6.5 GB | 96 GB | ~15x |
| GPU util | 36% | 100% | ~2.8x |

Substantial headroom remains. This matters for Phase 3: the TiledCamera thermal sensor is
expected to cut throughput significantly, and there is room to raise env count to compensate.
Baseline deliberately kept at Isaac Lab's tuned default of 4096 to avoid perturbing the
effective batch size (which would likely require LR retuning).

## Early learning signal (iteration ~33/1500)

| Metric | Value |
|---|---|
| Mean reward | -0.21 to -0.51 |
| Mean episode length | 67-86 steps |
| Episode_Termination/base_contact | 0.99 |

Robots fall almost immediately at this stage — expected for ANYmal rough terrain this early.

## Final result (iteration 1499/1500) — COMPLETE

| Metric | Value |
|---|---|
| Final mean reward | **17.71** |
| Mean reward @ iter 753 | 14.38 |
| Mean reward @ iter 1350 (pre-outage) | 16.27 |
| Exit status | `HARNESS_EXIT=0`, clean |
| Errors / OOM / CUDA faults | 0 |
| Final checkpoint | `logs/rsl_rl/anymal_c_rough/2026-08-13_07-56-55/model_1499.pt` |

Reward was still climbing at 1500, so this is a usable baseline rather than a converged
ceiling. Good enough as the Phase 3 comparison point; not worth more GPU time before the
deadline.

### Run was split across a pod outage

Iterations 0–1365 ran 2026-08-12 in `2026-08-12_14-32-14`; the pod became unreachable
mid-run. `/workspace` is a RunPod **network volume** (`mfs#eu-cz-1.runpod.net`), so the
67 GB venv, the Isaac Lab install, and every checkpoint survived intact — the outage cost
only the unfinished iterations, not the ~2 h rebuild a pod-local volume would have.

Resumed 2026-08-13 from `model_1350.pt` into `2026-08-13_07-56-55` with
`--resume --load_run 2026-08-12_14-32-14 --checkpoint model_1350.pt --max_iterations 150`.

Two things about resume worth knowing before reading these logs:

- **`max_iterations` is additive on resume, not absolute.** `OnPolicyRunner.learn` computes
  `total_it = start_it + num_learning_iterations` with `start_it` from the checkpoint, so
  150 means 1350 → 1500.
- **Mean reward restarts near zero and climbs back over ~40 iterations.** It is the mean
  over *completed* episodes, and resume resets every env, so the first episodes to finish
  are the early terminations. Reward went -0.02 → 15.25 over the first 34 iterations before
  passing its pre-outage value. This is not a lost policy; a freshly initialized policy does
  not reach 15 in 34 iterations.

## Phase 3 comparison — LiDAR-only rescue env (2026-08-13)

`Isaac-Rescue-Anymal-C-NoCam-v0`, same 4096 envs / 24 steps per env, 20 iterations measured.

| Metric | Baseline | Rescue (LiDAR, no camera) | Cost |
|---|---|---|---|
| Iteration time | ~2.40 s | ~3.07 s | +28% |
| Throughput | ~41,000 steps/s | **~32,000 steps/s** | -22% |
| Projected 1500 iters | ~53 min | **~77 min** | +24 min |

Clean exit, zero errors. The sensor suite minus the camera costs about a fifth of
throughput, so a full 1500-iteration rescue run fits comfortably before the deadline.

This directly contradicts review finding F7's 5-15x slowdown projection — correctly, since
that projection was driven by the thermal camera's 4096 x 64 x 64 segmentation render per
policy step, which this variant does not do. 288 LiDAR rays x 4096 envs of warp raycasting
is nearly free by comparison. F7's schedule warning still stands for the *camera* variant,
which has never been measured because it cannot start (see below).

### The thermal camera has never run

`--enable_cameras` aborts at launch on this pod: `vkCreateInstance` fails, no GPU devices.
CUDA is unaffected. Ruled out: driver version mismatch (kernel module and userspace both
580.173.02), missing graphics libraries, a malformed ICD manifest, and a wrong device node.
Untested best hypothesis: `/proc/driver/nvidia/gpus/` exposes all 8 host GPUs while only
`/dev/nvidia2` exists in the container; no `CAP_MKNOD` to test it.

### Separately: a SIGSEGV that was ours

Both rescue variants first died with exit 134/139 at what looked like Isaac Sim startup.
Root cause was `rescue_mdp.py` importing Isaac Lab's *runtime* classes (`Articulation`,
`RigidObject`, `Camera`, `RayCaster`) at module scope. Those cannot be imported before
`SimulationApp` exists, and hydra resolves the env cfg entry point before
`launch_simulation` — so the process died during config resolution while every traceback
pointed at `_create_app`. Fixed in 2f4c2d5 by moving them under `TYPE_CHECKING`, matching
what `isaaclab_tasks`' own mdp modules do.

## Notes

- `--headless` is deprecated in Isaac Sim 6.0; use `--viz none`.
- Isaac Sim 6.0 uses the Newton physics backend (pulls in `newton`, `mujoco-warp`).
- Missing X11 libs (`libxt6` et al.) caused non-fatal MaterialX load errors; installed.

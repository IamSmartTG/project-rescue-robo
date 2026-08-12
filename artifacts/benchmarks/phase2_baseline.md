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
Final values to be filled in on run completion.

## Notes

- `--headless` is deprecated in Isaac Sim 6.0; use `--viz none`.
- Isaac Sim 6.0 uses the Newton physics backend (pulls in `newton`, `mujoco-warp`).
- Missing X11 libs (`libxt6` et al.) caused non-fatal MaterialX load errors; installed.

# Phase 5 Benchmark — Hierarchical navigator (learned search)

Run dates: 2026-08-15 → 2026-08-16
Hardware: **NVIDIA Brev L40S, 46 GB VRAM** (massedcompute; `brev-ys88z1w2k` then `rescue-robo`)
Stack: Isaac Sim 6.0.1.0, Isaac Lab 3.0.0-beta2, RSL-RL PPO 5.0.1, Python 3.12

> **RESULTS PENDING** — this file is written up to the method. The eval numbers land after
> the resumed run finishes; every results cell below is marked `TBD` and must not be quoted
> until filled.

## What this phase is for

Phase 3 built the sensor suite and Phase 5 built the disaster scene, but the robot was never
*searching* either of them. `base_velocity` was still the inherited `UniformVelocityCommand`,
drawing a fresh random direction every 10 s. The thermal camera and LiDAR reached the
observation vector, and the victim rewards nudged the gait, but nothing let the policy use a
sensor to **choose a destination**. The Phase 5 debris numbers were therefore a baseline, not
a capability.

Two things close that loop, and both are measured here:

1. **The scripted seeker** (`seeker.py`) — thermal image → hottest pixels → bearing →
   velocity command. No training at all. A hand-written controller, and the writeup says so.
2. **The learned navigator** (`navigator.py` + `nav_env.py`) — an MLP that sees thermal and
   LiDAR and emits a *correction* to the seeker's command.

## The stack, and who runs at what rate

```
thermal ---------> scripted seeker ---> base command --+
                                                       +--> sum --> SHIELD --> walker
thermal + LiDAR --> navigator (MLP) ---> delta --------+                      (frozen)
```

| Layer | Rate | Trainable | Job |
|---|---|---|---|
| Navigator (this phase) | 10 Hz | **yes** | velocity-command delta from perception |
| Scripted seeker | 50 Hz | no | base command from the thermal centroid |
| LiDAR shield | 50 Hz | no | vetoes forward motion near an obstacle |
| Walker (`model_4197`) | 50 Hz | **frozen** | tracks whatever command it is handed |

Only the navigator has weights that move. Locomotion on rough terrain cost ~5200 iterations
across two stages and is not relearned — that is the entire reason this phase is affordable.

### Why the navigator runs at 10 Hz and not 50

`--inner_steps 5` decimates the 50 Hz sim to a 10 Hz decision rate. Acting every physics step
would need no inner loop and would be simpler, but with `gamma = 0.99` the effective horizon
is ~100 steps ≈ 2 s against a 20 s episode, so credit for reaching a casualty never propagates
back to the decision that turned toward it. Decimating by 5 stretches the same discount over
10 s. This is the ordinary reason hierarchical controllers run their high level slowly, not a
performance hack.

One navigator step accumulates reward over all 5 inner steps rather than taking the last —
the navigator owns everything that happened while its command was in force.

### Why a residual, and what it buys the comparison

The navigator adds a delta to the seeker's command rather than emitting the command outright.
With the output initialised near zero, **the policy at iteration 0 *is* the scripted seeker**
(Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019).

Two consequences worth stating plainly:

- Training can only move away from a known-decent controller, never from noise. The budget
  here affords roughly one run; if RL learns nothing, the result is still scripted-level.
- The baseline row and the treatment row come from **one code path** differing in exactly one
  flag (`use_residual`). The scripted-seeker number is not a different program — it is this
  program with the network switched off.

The useful part of the task (point at the warm pixels) is already known in closed form.
Spending the first thousand iterations rediscovering it would burn the budget on the part
that was never the research question.

### The shield is outside the network

`_apply_shield()` runs **after** the sum, so no amount of training can remove it. It vetoes
forward motion when any ray in a ±25° forward cone reads closer than 1.0 m, and deliberately
leaves the turn component alone — zeroing that too would freeze the robot against an obstacle
instead of letting it turn away. Reversing away stays allowed.

The ray indexing (288 rays = 8 channels × 36 azimuth steps, forward at index 18 because index
0 is −180°) carries a `_self_check()` in `navigator.py`, runnable without a simulator. An
off-by-one there shields the wrong direction and presents as "the policy refuses to walk",
which is a very expensive thing to debug inside a 4096-env training run.

**`shield_veto_rate` is the honesty check on this whole phase.** A navigator that scores well
while the shield fires constantly did not learn to avoid debris — it learned to lean on a
hand-written safety layer. `eval_rescue.py` reports the fraction of live steps on which the
shield vetoed, so the claim can be checked rather than assumed.

## Training

| | |
|---|---|
| Task | `Isaac-Rescue-Anymal-C-Nav-v0` |
| Envs | 4096 |
| Iterations | 300 |
| Throughput | ~2999 env-steps/s, ~33 s/iteration |
| Env steps at iter 244 | 24,084,480 |
| Walker | `anymal_c_rescue/2026-08-13_19-20-52/model_4197.pt`, frozen |

### The run was interrupted, and the writeup should not hide it

The first attempt reached iteration **244 of 300** before the host was lost. The checkpoint
watchdog syncs every 20 minutes, so `model_225.pt` survived and iterations 226–244 did not.
The run was resumed from `model_225.pt` on a rebuilt box the next morning
(`--resume_checkpoint`, with `--max_iterations` additive: 225 + 75 = 300).

Resuming restores optimizer state and the iteration counter alongside the weights, so this is
a continuation rather than a fine-tune from a cold optimizer. The gap is the ~19 iterations of
progress between the last sync and the loss of the host, which were re-run.

## Results

> `TBD` — filled from `artifacts/benchmarks/eval_navigator.json` once the run completes.

All rows: 256 episodes, 2 casualties, 4 debris props, identical protocol.

| Run | Task | Success | SPL | Fall rate | Shield veto |
|---|---|---|---|---|---|
| Walker alone (random command) | `Isaac-Rescue-Anymal-C-v0` | 0.1250 | 0.0789 | 0.719 | n/a |
| Scripted seeker (navigator @ iter 0) | `Isaac-Rescue-Anymal-C-Nav-v0` | 0.2109 | 0.1561 | 0.523 | TBD |
| **Navigator (`model_300`)** | `Isaac-Rescue-Anymal-C-Nav-v0` | TBD | TBD | TBD | TBD |

The bar that matters is **0.2109**, not 0.1250. Beating the random-command walker only proves
that pointing the robot at warm pixels helps, which the scripted seeker already established
for free. The navigator earns its keep only by beating the seeker *and* keeping the shield
veto rate low.

### Fall-rate split

| Run | Total | Near debris | Terrain only |
|---|---|---|---|
| Walker alone | 0.719 | 0.270 | 0.449 |
| Scripted seeker | 0.523 | 0.168 | 0.356 |
| **Navigator** | TBD | TBD | TBD |

The debris-adjacent column is the one the navigator is supposed to move: it is the fraction of
falls attributable to rubble the policy could have steered around.

## Caveats

- **`Metrics/success_rate` in the training log is velocity-command tracking, not casualty
  finding.** It reads ~0.84 on the walker while real rescue success is far lower. It is not a
  rescue number and must never be quoted as one. Rescue success comes only from
  `harness/eval_rescue.py`.
- `Episode_Termination/base_contact` sat at ~0.71 late in training — the robot still falls
  often on this terrain. The navigator changes where the robot goes, not how well it walks.
- Victim reward is easy to farm on flat ground; `Curriculum/terrain_levels` (0.85 at iteration
  244) must be read alongside any victim-reward figure. See `phase3_rescue.md` for the
  single-stage failure this pattern caused.
- One training run, one seed. No error bars. With 256 episodes the success-rate standard error
  is roughly ±0.026, so differences smaller than about 5 points are not resolvable here.

# Phase 5 Benchmark — Hierarchical navigator (learned search)

Run dates: 2026-08-15 → 2026-08-16
Hardware: **NVIDIA Brev L40S, 46 GB VRAM** (massedcompute; `brev-ys88z1w2k` then `rescue-robo`)
Stack: Isaac Sim 6.0.1.0, Isaac Lab 3.0.0-beta2, RSL-RL PPO 5.0.1, Python 3.12

> **Headline:** the learned navigator lost to the hand-written controller it was built on, for
> **two separate and separately diagnosed reasons** — a residual that was never initialised as a
> residual, and an objective that pays the navigator for tracking its own command. Fixing the
> first made the second visible. Underneath both, the real limit is locomotion: the walker has
> never trained on debris, and rubble alone costs 26 points of success rate.
>
> | | Success (n=1024) |
> |---|---|
> | Scripted seeker (no learning) | **0.1553** |
> | Learned navigator, init fixed | 0.0977 |
>
> Reading order: [Results](#results) → [Root cause #1](#root-cause-the-residual-was-never-a-residual)
> → [Root cause #2](#why-the-navigator-is-scored-on-tracking-its-own-command)
> → [What actually limits this robot](#what-actually-limits-this-robot).

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
The design intent is that with the output initialised near zero, **the policy at iteration 0
*is* the scripted seeker** (Johannink et al., *Residual Reinforcement Learning for Robot
Control*, ICRA 2019).

> ⚠️ **That intent was not implemented in the Aug 15 run.** The output head was left at default
> init, so iteration 0 was *not* the scripted seeker and the guarantee below did not hold. This
> section describes the design; [Root cause](#root-cause-the-residual-was-never-a-residual)
> describes what the code actually did. Everything from here to the end of this subsection is
> what the design *promises*, and is only true of the corrected run.

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
| Iterations | 300 (Aug 15 run), 100 (corrected run) |
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

The final checkpoint is **`model_299.pt`**, and no `model_300.pt` exists. RSL-RL's training loop is
`range(start_it, start_it + n)`, so a 300-iteration run ends on index 299 — that file *is* the
300th iteration, not one short of it. Worth stating because "where is model_300" is the
obvious first question and the obvious wrong answer is "the run was truncated".

Startup logged ~34 PhysX `collisionStackSize buffer overflow -- contacts have been dropped`
errors, all within the first second while 4096 environments spawn. They do not recur once the
sim is stepping. Noted rather than buried, since dropped contacts are in principle the same
signal fall detection reads.

## Results

All rows: 256 episodes, 2 casualties, 4 debris props, identical protocol. Reproduce the table
and the significance tests with `python3 harness/compare_evals.py`.

| Run | Success | ±SE | SPL | Fall | Fall near debris | Shield veto |
|---|---|---|---|---|---|---|
| Walker alone (random command) | 0.1250 | 0.021 | 0.0789 | 0.719 | 0.270 | n/a |
| Scripted seeker (Aug 15) | 0.2109 | 0.026 | 0.1561 | 0.523 | 0.168 | n/a |
| **Scripted seeker (Aug 16)** | **0.2500** | 0.027 | 0.1788 | 0.445 | 0.148 | 0.027 |
| Navigator iter 0 | 0.1289 | 0.021 | 0.0768 | — | — | — |
| Navigator iter 100 | 0.1328 | 0.021 | 0.0969 | 0.746 | 0.227 | 0.022 |
| Navigator iter 200 | 0.1289 | 0.021 | 0.0731 | 0.703 | 0.289 | 0.022 |
| Navigator iter 299 | 0.1602 | 0.023 | 0.1195 | 0.695 | 0.270 | 0.019 |

Two-sided z on proportions, against the scripted seeker (Aug 16):

| Comparison | Δ | z | p | |
|---|---|---|---|---|
| walker alone | −0.1250 | −3.67 | 0.0002 | significant |
| scripted seeker (Aug 15) | −0.0391 | −1.05 | 0.2930 | **not** significant |
| navigator iter 100 | −0.1172 | −3.41 | 0.0007 | significant |
| navigator iter 200 | −0.1211 | −3.54 | 0.0004 | significant |
| navigator iter 299 | −0.0898 | −2.53 | 0.0113 | significant |

**The learned navigator is significantly worse than the scripted controller, at every
checkpoint measured.** It was never ahead.

Read the second row before the others: the *same* scripted seeker, same protocol, measured on
two different days, scored 0.2109 and 0.2500. That gap tests at p=0.29 — it is noise. It is
also the reason every claim here carries a test rather than a comparison by eye.

### The mechanism

The navigator did not fail to learn. It learned the wrong thing, and the trajectory metrics
say which:

| | Seeker | Navigator (299) | |
|---|---|---|---|
| Mean path length | 6.03 m | **4.58 m** | navigator takes a straighter line |
| Mean time to find | 5.49 s | **4.16 s** | and gets there sooner |
| Mean episode length | 13.29 s | 9.05 s | but episodes end early |
| Fall rate | 0.445 | 0.695 | because it falls |

It bought directness with falls, and the falls cost more than the directness gained. The
walker is frozen, so the low level cannot adapt to a high level that commands more aggression
than the gait can absorb on rough terrain. Freezing the walker is what made this phase
affordable; it is also what caps it.

Two controls worth stating:

- **Shield veto is ~2% for both.** Neither controller is being rescued by the safety layer, so
  "it learned to avoid obstacles" and "the shield saved it" are distinguishable here, and the
  answer is that the shield is barely engaging in either case.
- **Terrain difficulty moved the wrong way for the navigator and it still lost.** Mean terrain
  level ended at 2.47 for the navigator against 3.39 for the seeker. Levels are seeded randomly
  and then moved by the curriculum *during* the eval, so this is a consequence of walking badly
  (demotion), not a starting handicap. The navigator was measured on **easier** terrain and
  still scored lower.

## Root cause: the residual was never a residual

`navigator.py` promises that at iteration 0 the policy behaves exactly like the scripted
seeker, so training can only move away from a known-decent controller. **Nothing in the code
implemented that promise.**

`init_std=0.3` in `NavigatorPPORunnerCfg` controls the *exploration noise*. Evaluation uses the
distribution **mean**, and the mean is whatever the untrained MLP emits. The actor's output
layer was left at PyTorch's default `Linear` init (`mlp.4`, measured `absmean=0.044`,
`max=0.089` — consistent with the default ±1/√128 bound).

Measured on the Aug 15 run's `model_0`, feeding the actor normalized observations:

| | delta vx | delta vy | delta wz |
|---|---|---|---|
| mean | +0.084 | −0.099 | +0.004 |
| mean abs | 0.148 | 0.176 | 0.127 |
| p95 abs | 0.353 | 0.426 | 0.311 |

Against a ±1.0 m/s command clamp that is **15% of the entire admissible command range**,
including a standing sideways bias. That is not a correction to the seeker; it is a corruption
of it. The consequence is the `navigator iter 0` row above: **0.1289**, statistically
indistinguishable from the random-command walker's 0.1250, and nowhere near the 0.2500 the
seeker actually scores.

So the correct reading of the 300-iteration run is not "residual RL does not work here":

> Training started ~12 points below its intended baseline, and PPO spent the entire budget
> partially digging back out — 0.129 → 0.133 → 0.129 → 0.160. That is real learning, spent
> almost entirely on undoing an initialisation bug.

### The fix, and how it was verified

Zero the output head's weight **and** bias at construction (`_zero_output_layer` in
`harness/train_navigator.py`), on fresh starts only — doing it after a resume would zero
trained weights and discard the run silently.

Verified against the installed rsl-rl 5.0.1 rather than assumed:

- `rsl_rl/models/mlp_model.py` — `forward(..., stochastic_output: bool = False)`; evaluation
  therefore takes the deterministic branch.
- `rsl_rl/modules/distribution.py` — `GaussianDistribution.deterministic_output()` returns the
  MLP output unchanged.

With the head at zero, delta is identically zero for *every* observation, so iteration 0
reproduces the scripted seeker exactly rather than approximately.

Measured effect, both checkpoints one PPO update in:

| | mean \|delta\| as % of the command clamp |
|---|---|
| Default init (Aug 15 run) | **15.01%** |
| Zero init (corrected run) | **0.86%** |

A note that cost some confusion: **`model_0.pt` is not the initial policy.** RSL-RL's loop
collects, updates, *then* saves inside the same iteration, so `model_0.pt` is one gradient step
in. The zeroed head reads `6.4e-3` there rather than exactly `0`.

## Corrected run: zero-initialised residual

100 iterations from a genuine seeker-equivalent start (verified: mean |delta| 0.86% of the
command clamp against the old run's 15.01%), evaluated at 1024 episodes against a matched
1024-episode seeker control.

Why 1024 and not 256: at n=256 the standard error on a success rate near 0.25 is 0.027, so the
standard error of a between-arm difference is ~0.037 and only a change of ~0.10 or larger is
detectable. The two scripted-seeker runs differing by 0.039 at p=0.29 is that limit in action.
At n=1024 the per-arm SE falls to ~0.013.

| n = 1024 | Scripted seeker | Zero-init navigator (iter 99) |
|---|---|---|
| **Success** | **0.1553** ± 0.0113 | **0.0977** ± 0.0093 |
| SPL | 0.1073 | 0.0658 |
| Fall rate | 0.5254 | **0.3809** |
| Falls near debris | 0.1826 | **0.1270** |
| Mean path length | 5.96 m | 3.36 m |
| Mean time to find | 6.06 s | 10.15 s |
| Mean episode length | 12.17 s | 14.76 s |
| Shield veto | 0.0326 | 0.0173 |

Δ success = −0.0576, z = −3.94, **p = 0.0001**. Fixing the initialisation did not make the
navigator competitive. It made it *worse*, and in an informative direction: the corrected
policy falls far less (0.38 vs 0.53), moves far less (3.36 m vs 5.96 m), survives longer
(14.76 s vs 12.17 s) and finds fewer casualties.

**It learned to stop searching and stay alive.**

### Why: the navigator is scored on tracking its own command

`nav_env.py` accumulates the **full environment reward** as the navigator's training signal.
That reward was written for the walker, and its dominant terms are `track_lin_vel_xy_exp` and
`track_ang_vel_z_exp` — which reward the walker for *following the velocity command*.

The navigator **emits that command**. So it can raise its own reward by issuing commands that
are trivially easy to follow — small, smooth, achievable — whether or not they lead anywhere
near a casualty. The objective is circular.

Final-iteration reward composition:

| Term | Corrected run (iter 99) | Aug 15 run (iter 299) |
|---|---|---|
| `track_lin_vel_xy_exp` | **0.4069** | 0.2320 |
| `track_ang_vel_z_exp` | **0.2853** | 0.2030 |
| `victim_found` | **0.0565** ↓ | 0.0780 |
| `victim_approach` | 0.0131 | 0.0140 |
| `victim_heading` | 0.0510 | 0.0312 |
| `Metrics/success_rate` (command tracking) | **0.9272** | 0.2221 |
| `Episode_Termination/base_contact` | 0.3951 | 0.6609 |
| **Mean reward** | **11.70** | 6.80 |

Read the last row against the `victim_found` row. Total reward nearly doubled while the reward
for actually finding casualties went **down**. Tracking terms total 0.692 against 0.121 for all
three victim terms combined — a **5.7× imbalance** — so an optimiser that ignores rescue
entirely is behaving correctly. `Metrics/success_rate = 0.9272` is the tell: the navigator set
its own exam and scored 93%.

This is specification gaming, not a training failure. The policy maximised the objective it was
given. The objective was wrong.

**The fix (identified, not yet tested):** score the navigator on the victim terms only, so the
layer that chooses *where to go* is not paid for how easy it made the walker's tracking job.
Note the trap — strip the locomotion terms entirely and nothing penalises falling any more, so
the navigator may swing back to commanding suicidal velocities. The reward needs the victim
terms plus a fall penalty, not simply the victim terms.

## What actually limits this robot

Both navigator failures are layered on top of a more basic problem, and the numbers name it:

| Same walker, same task | Success | Fall rate |
|---|---|---|
| Before debris was added to the scene | **0.3867** | 0.414 |
| After 4 debris props were added | 0.1250 | 0.719 |

Adding rubble cost 26 points of success and raised falls by 30. **The walker has never trained
on debris.** Phase 5 built the disaster scene and never retrained the locomotion policy for it,
so every number on the debris scene is a zero-shot result against obstacles the gait has never
encountered.

That, not the search layer, is the dominant limit. A ~50% fall rate caps everything above it:
the navigator can point perfectly at a casualty and it does not matter if the robot is on its
side. Ablations on the pre-debris scene agree that locomotion is where the losses are —
removing the payload moved falls 0.414 → 0.324, and disturbances 0.414 → 0.340.

### Debris fine-tune

`model_4197` resumed on the current rescue task (debris present, payload already narrowed to
2–6 kg) for 500 further iterations — 4197 → 4696, ~6.0 s/iteration, 49 min.

**It did not work.** All rows n=1024, scripted seeker driving:

| | Success | ±SE | Fall | Falls near debris | Falls terrain-only | SPL | Time to find |
|---|---|---|---|---|---|---|---|
| Aug 13 walker (control) | 0.1553 | 0.011 | 0.5254 | 0.1826 | 0.3428 | 0.1073 | 6.06 s |
| Debris-finetuned walker | 0.1592 | 0.011 | 0.4912 | 0.1719 | 0.3193 | 0.1153 | 5.08 s |

| Metric | Δ | z | p | |
|---|---|---|---|---|
| success rate | +0.0039 | +0.24 | 0.8085 | not significant |
| fall rate | −0.0342 | −1.55 | 0.1214 | not significant |
| falls near debris | −0.0107 | −0.63 | 0.5260 | not significant |
| falls terrain-only | −0.0235 | −1.13 | 0.2583 | not significant |

Every metric drifted the right way and not one of them clears noise.

### The hypothesis was weaker than it was presented, and this is the correction

The fine-tune was justified by `0.3867` (pre-debris) against `0.1250` (post-debris) — the same
walker, the same task id. Two problems with that argument, both visible before the run:

1. **It is a confounded comparison.** Those two evals are separated by several commits, not one:
   the debris addition, but also the thermal-render change, the payload narrowing to 2–6 kg, and
   the SPL / fall-split reporting change. Attributing the entire 26-point drop to rubble assumed
   a single-variable change that the git history does not support.
2. **The fall breakdown already argued against it.** Debris-adjacent falls are 0.18 against 0.34
   terrain-only. Rubble was never causing most of the falls, so "the gait has never seen debris"
   could not have been the dominant term in a 26-point drop.

The run is still worth having: it converts a plausible story into a measured negative. What it
rules out is that **500 iterations of fine-tuning recovers the debris penalty**. What it leaves
open is whether the penalty needs far more training (Phase 3's stage 2 took 1200 iterations to
install a new objective) or whether rubble makes the *task* harder — occluding casualties and
lengthening routes — rather than making the *gait* worse. The second is now the better bet, and
it is a search-and-planning problem, not a locomotion one.

## Demo videos

`artifacts/videos/rescue_demo_v6.mp4` is the deliverable: scripted seeker on the
debris-fine-tuned walker — the best-measured configuration — found the casualty, no fall,
clean take on try 3 of 10. Its `summary.json` records `navigator_checkpoint: null`, so the
video's provenance is checkable rather than asserted.

`rescue_demo_v5.mp4` was rendered from the buggy-init navigator (`model_299`) and is kept only
as the record of that arm. It should not be used to illustrate the system.

**On the thermal panel.** The v5 render shows a uniformly ambient thermal panel at every frame,
including the one where the voice log declares the casualty found, which looked like a repeat of
the Phase 5 thermal-render bug. It is not. v6 shows the casualty as a large bright region in the
16×16 policy view. The difference is the spawn annulus: v5 used `--victim_radius 5.0 6.5`, and
the thermal camera's far clip is 8.0 m, so at that range a 0.30 m casualty is at or below one
pixel. v6 used `3.5 5.0`. The sensor was always working — the earlier eval evidence said as much
(the seeker's only input is thermal and it beats the blind walker at p=0.0002) — but the demo
framing hid it.

A demo take is chosen by re-rolling seeds until the episode is worth showing, which selects a
*representative successful* episode, not a typical one. With a ~49% fall rate the typical episode
is not a rescue. The eval tables, not the video, describe typical behaviour.

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
- **Absolute success rates depend on `--num_envs`, so numbers from different `n` must not be
  quoted side by side.** The identical scripted seeker measured 0.2109 and 0.2500 at n=256 but
  0.1553 at n=1024; pooling the two n=256 runs (0.2305) against n=1024 tests at p=0.0006, so
  this is a systematic effect and not sampling noise. `num_envs` sets the terrain grid and env
  spacing, not merely the sample count. Every comparison in this document is therefore made
  *within* a fixed n, and the conclusion that the navigator loses to the seeker holds
  independently at both n=256 and n=1024. Cross-n comparisons — in particular the pre-debris
  0.3867, which is an n=256 number — indicate direction only, never a margin.

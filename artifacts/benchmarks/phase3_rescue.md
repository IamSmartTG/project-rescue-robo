# Phase 3/4 Benchmark — ANYmal-C Rescue Policy (full sensor suite)

Run dates: 2026-08-13
Hardware: **NVIDIA Brev L40S, 46 GB VRAM** (`mute-teal-zebra`, massedcompute)
Stack: Isaac Sim 6.0.1.0, Isaac Lab 3.0.0-beta2, RSL-RL PPO, Python 3.12

Why not the RunPod Blackwell box the Phase 2 baseline ran on: RunPod cannot create a Vulkan
instance, which kills the RTX renderer and therefore the thermal camera outright. Confirmed
provider-wide by a throwaway probe pod on a different machine and a different driver. See
`phase2_baseline.md` and the project memory note. The L40S is slower but can actually render.

## The three policies

| | Task | Sensors | Payload |
|---|---|---|---|
| **Baseline** | `Isaac-Velocity-Rough-Anymal-C-v0` | height scan only | no |
| **LiDAR-only** | `Isaac-Rescue-Anymal-C-NoCam-v0` | + 288-ray LiDAR | yes |
| **Full rescue** | `Isaac-Rescue-Anymal-C-v0` | + 16x16 pseudo-thermal camera | yes |

Payload = emergency supply modelled as randomized added mass (2–12 kg on a ~30 kg trunk)
plus added rotational inertia, so the gait has to generalize over what the robot is carrying.

## Throughput

Measured at 4096 envs x 24 steps per env = 98,304 env-steps per iteration.

| Policy | Hardware | Iteration time | Throughput | 1000 iters |
|---|---|---|---|---|
| Baseline | RTX PRO 6000 | ~2.40 s | ~41,000 steps/s | ~40 min |
| LiDAR-only | RTX PRO 6000 | ~3.07 s | ~32,000 steps/s | ~51 min |
| Full rescue (thermal 16x16) | L40S | ~7.5 s | ~13,100 steps/s | ~125 min |

The full-rescue column changes two variables at once — the thermal camera *and* a slower
GPU — so it is not a clean camera-cost measurement. The camera is clearly the dominant term
regardless: it forces the RTX renderer to produce a segmentation pass per policy step across
4096 envs, which raycasting does not.

### Thermal resolution is the whole budget

The first search run used a 64x64 thermal image. That is 4096 pixels against ~250 elements of
proprioception + height scan + LiDAR, so **94% of the observation vector was camera**, and
the policy collapsed: mean episode length fell 19–24 → 7.76 steps with 100% of episodes
ending in a base crash. Dropping to 16x16 (256 pixels, ~50% of the vector) fixed it. The
camera is also rendered at a quarter of the policy rate.

## Two-stage curriculum beats single-stage

Training locomotion and victim search simultaneously does not work here, and the numbers say
why cleanly.

| | Single-stage (killed at 632/1500) | **Two-stage (3199 iters)** |
|---|---|---|
| `Curriculum/terrain_levels` | 0.2455 | **3.7186** |
| `Episode_Reward/victim_found` | 0.2406 | 0.1238 |
| Mean episode length | 797.77 | 793.99 |
| Mean reward | 11.72 | 14.08 |

Read that table carefully: the single-stage run has the **higher** victim reward. That is the
finding, not a defect. It earned it on tier-0.25 terrain — effectively flat ground — while the
two-stage policy earns less of it on tier-3.72 terrain. Quoting `victim_found` without the
terrain level next to it would invert the conclusion.

The single-stage run learned to find victims and never learned to walk on anything: after 632
iterations its terrain level was still ~0.25, i.e. it never left the flattest tier. Victim
reward is easy to farm on flat ground, so the policy took that gradient and stopped climbing.

The two-stage split fixes the ordering:

- **Stage 1** (`Isaac-Rescue-Anymal-C-Locomotion-v0`, 2000 iters, 2048 envs): all sensors
  attached and feeding the observation, but victim reward weights set to zero. Pure
  locomotion. Clean exit, final mean reward 7.16, episode length 748, terrain level 2.6206.
- **Stage 2** (`Isaac-Rescue-Anymal-C-v0`, +1200 iters, 4096 envs): resume stage 1's
  checkpoint, switch the victim rewards on, and gate the approach reward on the thermal
  camera's FOV so the task is *search* — turn and scan — rather than walk-at-a-victim-you-
  could-always-see.

The observation space is identical between stages, which is the entire reason stage 1 keeps
the sensors bolted on: the checkpoint loads straight into stage 2 with no surgery.

Env count changed 2048 → 4096 between stages. 4096 is the batch size
`AnymalCRoughPPORunnerCfg`'s learning rate, `desired_kl` and `num_mini_batches` were tuned
against, so 2048 was the mismatched setting, not the other way round. Changing env count does
not touch the policy network, so the checkpoint still loads.

`max_iterations` is additive on resume: `total_it = start_it + n`, so stage 2's `1200` means
1999 → 3199.

## Stage 2 final result

Run `anymal_c_rescue/2026-08-13_16-24-03`, final checkpoint `model_3198.pt`, 2 h 15 m wall
clock, 117,964,800 total env-steps.

| Metric | Value |
|---|---|
| Final iteration | 3198/3199 |
| Mean reward | **14.08** |
| Mean episode length | 793.99 steps (~15.9 s) |
| `Curriculum/terrain_levels` | **3.7186** |
| `Episode_Reward/victim_found` | 0.1238 |
| `Episode_Reward/victim_approach` | 0.0094 |
| `Episode_Reward/victim_heading` | 0.0405 |
| `Metrics/success_rate` (velocity tracking) | 0.8433 |
| `Episode_Termination/time_out` | 0.6600 |
| `Episode_Termination/base_contact` | 0.3400 |
| Errors / OOM / CUDA faults | 0 |

Two honest caveats on this table:

- **`Metrics/success_rate` = 0.8433 is velocity-command tracking, not victim finding.** It says
  the robot follows its commanded gait 84% of the time. It is not a rescue success rate and
  must not be quoted as one. The real number comes from the eval sweep below.
- **`stage2_search.sh` does not stamp `HARNESS_EXIT`**, unlike `run_train.sh`. "Clean exit"
  here means the log reaches 3198/3199 with zero tracebacks and the trainer released the GPU
  — not that an exit code was recorded. Worth fixing in the harness.

Terrain level was still climbing at the end (3.66 → 3.72 over the last ~150 iterations), so
this is a usable policy rather than a converged ceiling.

## Eval sweep — actual victim-find success rate

`harness/eval_rescue.py` runs the frozen policy over 256 parallel episodes with no learning
and counts what happened, because every number above is a training-time average taken while
the policy and the terrain curriculum were both still moving.

Full results: `artifacts/benchmarks/eval_rescue.json`. Checkpoint `model_3198.pt`, 256 episodes.

| Metric | Value |
|---|---|
| **Victim-find success rate** | **0.3867** |
| Mean time to first contact (successes only) | 5.49 s |
| Mean episode length | 759.1 steps (15.18 s) |
| **Fall rate** | **0.4141** |
| Mean terrain level evaluated on | 3.434 |

One thing that number is **not**: it is not measured on tier-3.72 terrain. A fresh env seeds
each robot's terrain level randomly up to `max_init_terrain_level` and lets the curriculum
move it from there — it does not inherit the level the training run climbed to. The eval ran
at tier 3.434, close enough to make the comparison fair, but it is measured, not assumed.

### What the eval says to fix next — and it is not the sensor

**41% of episodes end with the robot on its face.** That is the dominant failure mode, and it
caps victim-find success far more tightly than any perception gap: an episode that ends in a
fall at 6 s cannot find anything in the 10 s it no longer has. Success is 0.3867 while falls
are 0.4141 — the two numbers are close enough that locomotion robustness, not search
behaviour, is the binding constraint.

This directly contradicts the intuitive fix. The obvious thing to reach for is the sensor
blind spot below, because "the robot can't see the victim" is a satisfying story. The data
says the robot mostly *falls over* before the question of seeing arises. Eval fall rate
(0.4141) also runs worse than the training-time `base_contact` (0.3400), which is what you
would expect when a frozen policy meets terrain the curriculum is no longer easing for it.

Priority order the measurement implies:

1. **Fall rate.** Biggest lever on success by a wide margin. More training is the cheap first
   attempt (the extended run tests exactly this); reward shaping on base contact is the next.
2. **`detect_range` 10 m vs thermal 8 m.** Correctness fix, small expected effect on success.
3. Thermal resolution / FOV. No evidence yet that either is binding.

### Experiment: freeze the terrain curriculum — mostly a negative result

Hypothesis: `mdp.terrain_levels_vel` promotes a robot whenever it tracks velocity well, so by
construction it parks every robot at the edge of its competence and holds it there. A ~40%
fall rate is not a bug in that design, it *is* that design. Freezing the curriculum should let
the policy consolidate at a stationary difficulty and fall less.

Run `anymal_c_rescue/2026-08-13_19-20-52`, 1000 iterations resumed from `model_3198.pt` with
the hydra override `env.curriculum.terrain_levels=null`. Override confirmed live, not merely
typed: the log reports `<CurriculumManager> contains 0 active terms` and `terrain_levels`
appears 0 times, against 1201 times in `stage2.log`.

| Metric | Stage 2 (curriculum on) | Frozen curriculum | Change |
|---|---|---|---|
| Fall rate | 0.4141 | **0.3906** | −0.0235 |
| Victim-find success | 0.3867 | **0.3867** | none |
| Mean time to find | 5.49 s | 4.63 s | −0.86 s |
| Mean episode length | 759.1 | 780.6 | +21.5 |
| Terrain tier evaluated on | 3.434 | 3.359 | comparable |

**Read this as a near-null result, not a win.** Fall rate moved 2.4 points on 256 episodes —
around the noise floor for that sample size — and success rate did not move at all. 1000
iterations of consolidation did not meaningfully stabilise the robot, which means the
curriculum is not the dominant source of falls. The honest conclusion is that the hypothesis
was largely wrong.

What that leaves, in order of what the evidence now supports:

1. **The disturbance events.** Tested — see below.
2. **Reward weights.** No penalty scales with base contact beyond termination itself.
3. Terrain generator mix — the rubble tiers may simply be beyond an 0.30 m-clearance robot.

### Experiment: turn the disturbances off — the falls are the terrain

`harness/eval_rescue.py --no_disturbance` nulls `push_robot` and
`base_external_force_torque` and changes nothing else — same policy, same terrain, same
payload, same observation noise. Switching to the PLAY task would have disabled them too but
shrunk the terrain grid at the same time, confounding the variable.

| Metric | Disturbances on | Disturbances off | Change |
|---|---|---|---|
| Fall rate | 0.3906 | **0.3398** | −0.0508 |
| Victim-find success | 0.3867 | 0.3633 | −0.0234 |
| Mean episode length | 780.6 | 805.3 | +24.7 |

**The shoves account for about 5 points of a 39-point fall rate. The other 34 points are the
robot failing on the terrain unaided.** That settles the question the two experiments were
built to answer: this is a locomotion-capability problem, not an artefact of training
randomisation and not a curriculum-scheduling artefact.

Success rate drifting *down* slightly with disturbances removed is almost certainly noise at
n=256 — it should be ignored rather than explained. If it survives a larger sample it would
be interesting, but nothing here justifies a story about it yet.

### Experiment: drop the emergency-supply payload — the largest single contributor

`--no_payload` nulls `add_payload_mass` and `add_payload_inertia`, nothing else. Note this
measures an unloaded robot the policy was never trained as: it trained always carrying
something, so a *rise* in falls would have meant the policy had come to rely on the mass.

| Metric | Payload on (2–12 kg) | Payload off | Change |
|---|---|---|---|
| Fall rate | 0.3906 | **0.3242** | −0.0664 |
| Victim-find success | 0.3867 | **0.4062** | +0.0195 |
| Mean time to find | 4.63 s | 6.32 s | +1.69 s |

The payload is the biggest single contributor found — larger than the disturbances and the
curriculum combined — and it is the only intervention that moved success rate in the right
direction as well as fall rate.

### Fall-rate budget, all three experiments

Starting from 0.3906 on the frozen policy, each tested in isolation:

| Removed | Fall rate | Points recovered |
|---|---|---|
| (nothing) | 0.3906 | — |
| Terrain curriculum scheduling | 0.4141 → 0.3906 | 2.4 |
| Disturbance events | 0.3398 | 5.1 |
| **Emergency-supply payload** | **0.3242** | **6.6** |
| Floor: terrain vs. robot capability | ~0.32 | ~32 unexplained |

Even with every removable handicap gone, roughly a third of episodes still end in a fall.
That residual is the robot against the terrain, and it is the real ceiling on this policy.

**The payload is not a bug to delete** — it is a project requirement, the most defensible
"rescue" element in the training, and the thing that makes the gait generalise over what the
robot is carrying. The actionable version is to narrow the range (2–12 kg is a 7–40% swing on
a ~30 kg trunk; 2–6 kg would halve it) or to train substantially longer with it, rather than
to drop it and quote the better number.

Recommended order for whoever picks this up:

1. Narrow the payload range and retrain. Best measured evidence, and it keeps the requirement.
2. More iterations — this policy has seen only ~5,200 total, and reward was still climbing.
3. Terrain generator mix against ~0.30 m ground clearance.
4. `detect_range` 10 m → 8 m (correctness, unrelated to falls).

The modest gains that *did* appear (time-to-find −0.86 s, episode length +21.5 steps) are
consistent with a slightly more settled gait, which is worth keeping. The frozen checkpoint
`model_4197.pt` is therefore the one used for the demo video.

### Known blind spot, measured not guessed

The thermal camera sits 0.22 m above the trunk (~0.79 m up) pitched 15 deg down, with a
30 deg vertical half-FOV — so it covers 15 deg above horizontal to 45 deg below. A 0.30 m
casualty leaves that cone only inside `atan(0.64/d) > 45 deg`, i.e. **closer than 0.64 m**,
and `found_radius` is 1.0 m, so detection fires before the robot goes blind. That near blind
spot is already engineered out (the 15 deg pitch and the raised mount were both measured
fixes — a level 0.05 m mount spent 74% of every frame imaging the robot's own chassis).

What is **not** consistent: `victim_approach` uses `detect_range = LIDAR_MAX_RANGE = 10.0 m`,
while the thermal camera's far clip is 8.0 m and the FOV gate is computed from geometry
rather than from rendered pixels. In the 8–10 m band the policy is therefore paid for closing
on a casualty the thermal sensor cannot see — the exact clairvoyance the gate exists to
prevent. Victims spawn at 1.6–3.6 m so the band is only reachable mid-episode, which is why
this was not caught earlier.

Fix is one value (`detect_range` → 8.0). Deliberately **not** applied before the extended run,
so that run stays a clean continuation and the eval number above measures the policy that was
actually shipped. Apply it with the eval as the before/after evidence.

## Reward curves

TensorBoard event files sync to `artifacts/checkpoints/anymal_c_rescue/<run>/`. Read them with:

```
tensorboard --logdir artifacts/checkpoints/anymal_c_rescue
```

No static plots are committed — the event files are the source of truth and TensorBoard
already renders them.

## Demo video — three sensor views

`artifacts/videos/rescue_demo_v2.mp4` (and `_v3` if a clean take was found), 1640x720:

```
+---------------------------+--------+
|                           | THERMAL|   chase   1280x720
|      CHASE CAM 1280x720   | 360x360|   thermal  360x360   16x16 upscaled, ironbow
|                           +--------+   lidar    360x360   bird's-eye scope
|                           | LIDAR  |
+---------------------------+--------+
```

The LiDAR panel plots hits rotated into the robot frame, coloured by height rather than
range — the fan is tilted down, so on open ground a range-coloured scope is a bland ring of
floor returns, while height separates steps, rubble and casualties from the floor itself.
Misses come back as `inf` and are dropped rather than drawn at max range, which would paint a
fake wall around the robot.

**Recorded on `Isaac-Rescue-Anymal-C-Play-v0`**, which disables `push_robot`,
`base_external_force_torque` and observation corruption. That is the standard Isaac Lab
inspection variant and normal practice for a demo, but it is stated here rather than implied:
the robot is visibly steadier on video than the 0.39 eval fall rate would suggest, and that
gap is the disabled disturbances, not extra skill.

The renderer writes `summary.json` per take (`found_victim`, `fell`) and re-rolls the seed
looking for a clean run, so the video is a representative successful episode rather than
whichever episode luck dealt — also stated rather than implied.

## Earlier two-panel demo

`artifacts/videos/` — chase camera parented to the robot's base prim, with the 16x16 thermal
view composited beside it and the voice-comms caption burned in on casualty contact.

Isaac Lab's stock `--video` records the *static viewport* camera, so a walking robot leaves
frame within seconds and the clip is mostly empty terrain. `ViewerCfg.origin_type="asset_root"`
would track it but is deprecated and silently ignored in 6.0. `harness/render_chase.py` sidesteps
the viewport entirely by bolting an RGB `CameraCfg` to the robot, exactly like the thermal sensor.

One renderer gotcha, since it cost real time: that chase camera renders **pure black** in any
task that also carries the thermal camera, while working fine in the camera-free task. The RTX
renderer only produces a colour AOV when some camera asks for one, and the thermal camera asks
only for `semantic_segmentation`. Adding `rgb` to the thermal camera's `data_types` fixes it.

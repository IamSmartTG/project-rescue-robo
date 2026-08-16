# Mistakes

A record of what went wrong in this project and what each one cost. Kept because the expensive
mistakes here were not coding errors — they were *reasoning* errors that produced working code,
plausible numbers, and confident write-ups. Those are the ones that survive review.

Ordered by cost, not by chronology.

---

## 1. A guarantee that was documented but never implemented

**What.** `navigator.py` states that with the output initialised near zero, the policy at
iteration 0 *is* the scripted seeker, so training can only improve on a known-decent controller.
Nothing in the code did that. `init_std=0.3` in `NavigatorPPORunnerCfg` shrinks the *exploration
noise*; evaluation uses the distribution **mean**, and the actor's output head sat at PyTorch's
default `Linear` init.

**Cost.** The entire 300-iteration navigator run, plus the 100-iteration rerun that found it.
Measured: the untrained head emitted 15.01% of the ±1.0 command clamp with a standing sideways
bias, so iteration 0 scored **0.129** — the random-command walker's number — instead of the
seeker's 0.250. Training spent its whole budget climbing out of that hole (0.129 → 0.133 →
0.129 → 0.160) and was reported as "RL fails to beat the baseline".

**Why it survived.** The docstring was detailed, correct in intent, and cited the right paper.
It read like a description of the code. Nobody checks a claim that is already written down
confidently.

**Lesson.** A design guarantee stated in prose is a hypothesis about the code until something
executes it. `_zero_output_layer` now enforces it and prints the layer shape; the run aborts if
the head is not where it is assumed to be.

---

## 2. Training a controller on a reward for following its own orders

**What.** `nav_env.py` accumulates the **full environment reward** as the navigator's training
signal. That reward was written for the walker, and its dominant terms (`track_lin_vel_xy_exp`,
`track_ang_vel_z_exp`) pay for *tracking the velocity command* — which is precisely what the
navigator emits.

**Cost.** The corrected run produced the best training curve in the project (mean reward 6.80 →
11.70) and the *worst* rescue performance (0.0977 against the seeker's 0.1553, p=0.0001). The
navigator learned to issue easy-to-follow commands and stop searching: path length 3.36 m
against 5.96 m, time-to-find 10.15 s against 6.06 s.

**The tell that was ignored for hours.** `Metrics/success_rate` hit **0.9272**. That metric is
velocity-command tracking, and the project already knew it was not a rescue number — it is
written down in `phase3_rescue.md` and in the memory notes. A controller scoring 93% on the
metric it *controls the difficulty of* is the signature of a circular objective, and it was
visible in every training log before any eval was run.

**Lesson.** When an agent's action determines the difficulty of the thing it is scored on, the
objective is circular. Check that the reward measures an outcome the policy cannot trivially
make easy.

---

## 3. Comparing across a confounded change, then spending GPU on the conclusion

**What.** The claim "rubble costs 26 points of success" came from 0.3867 (before debris) against
0.1250 (after). Those two evals differ by more than debris: several commits apart, and the
casualty count went from **three to two**. `rescue_env_cfg.py` says so in as many words —

> Expect success rate to fall against the 0.3867 measured on three casualties with no debris.
> That is the task getting harder, and it must be reported as a separate row rather than
> compared directly.

It was compared directly anyway, presented as the project's limiting factor, and used to justify
a 500-iteration fine-tune.

**Cost.** 49 minutes of GPU and about $1, on a hypothesis the codebase had already warned
against. The fine-tune moved success by +0.0039 (p=0.81) — nothing.

**A second signal that was already in hand.** Debris-adjacent falls were 0.18 against 0.34
terrain-only. Rubble was never causing most of the falls, so it could not have been the dominant
term in a 26-point drop. The number was in the eval JSON before the run was launched.

**Lesson.** Before spending compute on a comparison, check what else changed between the two
measurements — and read the warnings already written in the config.

**Postscript, and it cuts both ways.** The single-variable experiment was eventually run —
one checkpoint, one session, toggling only `--no_debris` — and rubble *does* cost a lot:
success 0.1602 → 0.2266 (p=0.0001), fall rate 0.5254 → 0.2754. So the instinct was right and
the *evidence offered for it* was wrong, which is a distinction worth keeping. The "26 points"
figure remained wrong (it also contained the casualty count going 3 → 2); the real
single-variable cost is 6.6 points. And the fine-tune still failed — not because the penalty
was imaginary, but because 500 iterations cannot learn it.

The order was the actual error: **the $1 measurement should have come before the $1
fine-tune.** A/B-ing a flag on an existing checkpoint is nearly always cheaper than training
anything, and it is the thing that tells you whether training is worth it.

---

## 4. `num_envs` changes the measurement, not just the sample size

**What.** The same scripted seeker, same protocol, same checkpoint: **0.2109** and **0.2500** at
n=256, **0.1553** at n=1024. Pooling the n=256 runs against n=1024 tests at p=0.0006, so this is
systematic, not sampling noise. `num_envs` sets the terrain grid and env spacing.

**Cost.** Every cross-`n` comparison in the earlier write-ups was invalid and had to be
re-scoped. Worse, it nearly produced a false headline: the navigator's 0.1602 at n=256 looks
competitive against the seeker's n=1024 number, and would have read as "roughly matched".

**Lesson.** Fix `n` per comparison and state it on every chart. Conclusions here hold only
because they were re-derived within each `n` independently.

---

## 5. Artifacts that could not prove what produced them

**What.** `render_chase.py` wrote a `summary.json` recording the walker checkpoint but not the
navigator one, and `make_demo.sh` filtered the "navigator loaded" line out of its log. On the
Nav task, a render with no navigator silently films the **scripted seeker** — and looks
identical.

**Cost.** `rescue_demo_v5.mp4` existed for hours with no way to establish which policy drove it,
during a period when the navigator's performance was the entire question.

**Lesson.** An artifact that cannot name its own inputs is not evidence. `summary.json` now
records `navigator_checkpoint` (explicitly `null` when absent) and `inner_steps`.

---

## 6. A checkpoint watchdog slower than the failure it guards against

**What.** The sync watchdog ran on a 20-minute cycle. The training host was reclaimed at
iteration 244 of 300.

**Cost.** Iterations 226–244 and the final checkpoint. Recovering them needed a full stack
rebuild the next morning and a resume path that did not exist yet.

**Lesson.** The backup interval is the size of the loss when the thing you fear happens. On the
rerun it was cut to 10 minutes, and the corrected run wrote to a **separate** log directory —
see mistake 7.

---

## 7. Near-miss: the rerun would have deleted the evidence for its own finding

**What.** The corrected run was about to write to `logs/rsl_rl/navigator/`, reusing
`model_0/25/50/75`. The watchdog pulls remote over local by mtime, so it would have silently
overwritten the local `model_0.pt` and `model_100.pt` — the exact checkpoints proving the
initialisation bug. The eval JSONs would have survived and become unreproducible.

**Cost.** None. Caught in review before launch, and only because a review was requested.

**Lesson.** Recorded as a near-miss precisely because nothing went wrong. A fresh run reusing a
previous run's directory is a silent-overwrite hazard whenever a sync job is watching it.

---

## 8. Smaller ones worth not repeating

- **`isaaclab.sh` returns 0 after a Python traceback.** Two curve evals were launched against
  checkpoints that had never been pushed to the new host, died on `FileNotFoundError`, and
  reported `EXIT=0`. ~20 minutes of GPU. Check for the *output file*, never the exit code.
- **`model_0.pt` is not the initial policy.** rsl-rl collects, updates, *then* saves inside the
  same iteration. A zero-init verification against `model_0.pt` reads `6.4e-3`, not `0`, and
  looks like the fix failed.
- **`isaaclab.sh` picks its interpreter from `VIRTUAL_ENV`.** A non-interactive SSH does not
  inherit it, so it silently used system Python 3.10 and died on a stdlib import (`tomllib`).
  Every remote command needs `source ~/isaac/venv/bin/activate` first.
- **A global `sed` rewrote a sentence into a tautology** ("`model_299.pt`, not `model_299.pt`").
  Search-and-replace across prose needs the diff read afterwards.
- **`brev delete` does not always take on the first call.** Deleting `rescue-debris` returned
  the instance name and exited cleanly, and the instance then showed `DEPLOYING` — not
  `DELETING` — on the next poll. A second delete was needed. The command's success output is
  not proof the machine is gone; only `brev ls` reporting no instances is. This is the exact
  shape of the failure that once left an L40S running ~17 hours and took the account negative,
  so verify in a loop rather than trusting the return.
- **The thermal panel was called a probable bug before the cheap explanation was checked.** It
  was correct all along: `--victim_radius 5.0 6.5` against an 8.0 m far clip puts a 0.30 m
  casualty at or below one pixel. Investigating was right; asserting a likely cause first was
  not.

---

## The pattern

Six of these produced **no error message**. The code ran, the numbers looked reasonable, and the
write-up was confident. What caught them was cheap measurement of things assumed to be true — the
delta magnitude at initialisation, the reward composition per term, the same policy measured
twice, the checkpoint's actual weights.

The most expensive habit in this project was treating a well-written comment as a description of
the code.

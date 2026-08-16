# Presentation pack — Rescue Robot, hierarchical navigator

Everything needed to build the deck: per-slide message, the exact figures, what to chart, and
speaker notes. Numbers come from `artifacts/presentation/eval_data.{csv,json}`, generated from
the eval JSONs — do not retype them by hand.

**The one rule that will bite you:** never put an `n=256` number next to an `n=1024` number.
The same controller scores 0.2109 / 0.2500 at n=256 and 0.1553 at n=1024 (p=0.0006). Env count
changes the terrain layout, not just the sample size. Each chart must fix one `n` and say so.

**Story in one line:** we made the sensors decide where the robot goes, and measuring it
honestly showed the learned layer losing to a 40-line hand-written controller — for two
separate reasons we found and can name, on top of a locomotion limit that caps everything.

---

## Slide 1 — Title

**Rescue Robot: teaching a quadruped to search a disaster scene**
ANYmal-C · Isaac Sim 6.0.1 · Isaac Lab 3.0 · PPO · 2026-08-12 → 08-16

Speaker note: five days, from a locomotion baseline to a sensor-driven search stack, plus an
honest measurement of what works.

---

## Slide 2 — The problem

A collapsed structure. A quadruped with a thermal camera, a 288-ray LiDAR and an emergency
supply payload. It has 20 seconds to reach a casualty.

**Visual:** one frame from `artifacts/videos/rescue_demo_v5.mp4` — chase cam + thermal panel +
LiDAR scope. Extract with:
`ffmpeg -ss 3.7 -i rescue_demo_v5.mp4 -frames:v 1 hero.png`

Speaker note: the three panels are the whole argument that the sensors are load-bearing — left
is what an observer sees, right is what the robot senses.

---

## Slide 3 — The stack (architecture diagram)

Draw this as four stacked boxes, one highlighted as the only trainable one:

```
thermal ---------> scripted seeker ---> base command --+
                                                       +--> sum --> SHIELD --> walker
thermal + LiDAR --> navigator (MLP) ---> delta --------+                      (frozen)
```

| Layer | Rate | Trainable |
|---|---|---|
| Navigator (MLP) | 10 Hz | **yes** |
| Scripted seeker | 50 Hz | no |
| LiDAR safety shield | 50 Hz | no |
| Walker `model_4197` | 50 Hz | **frozen** |

Speaker note: only the top layer learns. Locomotion took ~5200 iterations across two stages and
is not relearned — that is the entire reason this phase was affordable at all.

---

## Slide 4 — Two design decisions worth defending

1. **The navigator runs at 10 Hz, not 50.** With `gamma=0.99` the effective horizon is ~2 s
   against a 20 s episode, so credit for reaching a casualty never reaches the decision that
   turned toward it. Decimating by 5 stretches the discount over 10 s.
2. **The shield sits outside the network.** It vetoes forward motion inside a ±25° cone at
   1.0 m, and no amount of training can remove it. Turning is deliberately left alone — zeroing
   that too would freeze the robot against an obstacle instead of letting it turn away.

Speaker note: the shield being untrainable is what lets us *test* whether the policy learned
obstacle avoidance, rather than assume it.

---

## Slide 5 — Result: the learned layer lost

**Chart: horizontal bar, success rate, n=1024 ONLY.** Two bars. Error bars = ±1 SE.

| | Success | ±SE |
|---|---|---|
| Scripted seeker (no learning) | **0.1553** | 0.0113 |
| Learned navigator (init fixed) | 0.0977 | 0.0093 |

Δ = −0.0576, z = −3.94, **p = 0.0001**

Speaker note: this is the headline and it is a negative result. Say it plainly, then spend the
rest of the deck showing that we know exactly why — which is the actual contribution.

---

## Slide 6 — It didn't fail randomly. It learned to stand still.

**Chart: grouped bars, normalised, n=1024.** Seeker vs navigator across five metrics. Colour the
"navigator better" bars differently from "navigator worse" — the point is that it wins on
safety and loses on the mission.

| | Seeker | Navigator | |
|---|---|---|---|
| Success | 0.1553 | 0.0977 | worse |
| Fall rate | 0.5254 | **0.3809** | better |
| Path length | 5.96 m | 3.36 m | barely moves |
| Time to find | 6.06 s | 10.15 s | slower |
| Episode length | 12.17 s | 14.76 s | survives longer |

Speaker note: it falls far less and survives far longer. It achieved that by not going
anywhere. That is a clue, not a mystery.

---

## Slide 7 — Root cause #1: the residual was never a residual

The design says the policy starts *identical* to the scripted controller, so training can only
improve on it. Nothing in the code implemented that. `init_std=0.3` shrinks exploration noise;
evaluation uses the **mean**, and the output layer sat at PyTorch default init.

**Chart: bar, "mean |delta| as % of the ±1.0 command clamp".**

| | % of command range |
|---|---|
| Default init (as run) | **15.01%** |
| Zero init (fixed) | **0.86%** |

Consequence, n=256: navigator at iteration 0 scored **0.1289** — statistically identical to the
*random-command* walker (0.1250), not the seeker's 0.2500.

Speaker note: the run wasn't "RL failing to beat a baseline". It started 12 points below the
baseline it was supposed to *start at*, and spent 300 iterations digging out.

---

## Slide 8 — Root cause #2: the objective was circular

The navigator emits the velocity command. The reward pays the walker for **tracking that
command**. The navigator is trained on that same reward. So it can raise its own score by
commanding velocities that are easy to follow — regardless of where they go.

**Chart: stacked bar, reward composition at the final iteration.**

| Term | Corrected run | Old run |
|---|---|---|
| `track_lin_vel_xy_exp` | 0.4069 | 0.2320 |
| `track_ang_vel_z_exp` | 0.2853 | 0.2030 |
| all three victim terms | **0.121** | 0.123 |
| **mean reward** | **11.70** | 6.80 |
| `Metrics/success_rate` (tracking) | **0.9272** | 0.2221 |

Tracking outweighs the actual mission **5.7×**. Total reward nearly doubled while
`victim_found` went **down** (0.0565 vs 0.0780).

Speaker note: `Metrics/success_rate = 0.9272` is the smoking gun — the navigator set its own
exam and scored 93% on it. This is specification gaming. The optimiser was right; the objective
was wrong.

---

## Slide 9 — What actually limits the robot

**Chart: two bars, n=256, same walker, same task.**

| | Success | Fall rate |
|---|---|---|
| Before debris existed in the scene | **0.3867** | 0.414 |
| After 4 debris props added | 0.1250 | 0.719 |

The walker has **never trained on debris**. Every debris-scene number is zero-shot against
obstacles the gait has never seen. Rubble alone costs 26 points.

Speaker note: a ~50% fall rate caps everything above it. The search layer can point perfectly
at a casualty and it does not matter if the robot is on its side. This is the highest-value
thing to fix, and it is not a search problem.

---

## Slide 10 — Debris fine-tune result

> **FILL FROM** `artifacts/benchmarks/eval_debris_finetune_n1024.json` once the run completes.
> Chart: three bars at n=1024 — seeker 0.1553, zero-init navigator 0.0977, retrained walker ?

Speaker note: state whether it moved and by how much. If it did not clear the seeker, say so —
the diagnosis stands on the 0.3867 evidence either way.

---

## Slide 11 — How the numbers were kept honest

Four practices worth one slide, because they are why the result is trustworthy:

1. **Every claim carries a significance test.** The same controller measured 0.2109 and 0.2500
   on two runs (p=0.29). Eyeballing 256-episode results is not safe.
2. **Sample size was raised when the question got finer.** n=256 → n=1024 once we needed to
   resolve differences below 10 points.
3. **A falsifiable control for every claim.** `shield_veto_rate` (~2–3% both arms) distinguishes
   "learned to avoid obstacles" from "the safety layer saved it".
4. **Negative results kept.** The retracted thermal render, the init bug, and this result are
   all in the record.

Speaker note: the eval harness reads `found_radius` and the victim list off the reward term
itself, so success rate and SPL can never describe a different task than the reward does.

---

## Slide 12 — What we would do next

1. **Retrain locomotion on debris** — largest measured gap (26 points), evidence on slide 9.
2. **Give the navigator its own reward** — victim terms plus a fall penalty, never the walker's
   tracking terms. (The trap: strip tracking entirely and nothing penalises falling, so it will
   swing to commanding suicidal velocities.)
3. **Add memory.** Both controllers are stateless — a casualty that goes behind rubble is
   forgotten instantly. This is what a learned navigator should beat a scripted one at, and it
   was never the thing being tested.

---

## Appendix slides (keep in the back pocket)

**A1 — Full results table.** All nine rows from `eval_data.csv`, grouped by `n`, with the `n`
column visible so nobody cross-reads.

**A2 — Two-stage curriculum (Phase 3).** Single-stage got the *higher* victim reward (0.2406 vs
0.1238) by farming casualties on flat ground — `terrain_levels` stuck at 0.2455 vs 3.7186.
Lower victim reward, far better robot. Good slide if asked "how do you know the reward is
measuring the right thing".

**A3 — Thermal resolution.** 64×64 made 94% of the observation vector camera and collapsed
episode length 19–24 → 7.76 steps. 16×16 fixed it.

**A4 — Infrastructure honesty.** Two cloud instances reclaimed within twelve hours, one
mid-run; a checkpoint watchdog turned a lost host into ~10 lost minutes instead of a lost day.

---

## Chart build notes

- **Fix `n` per chart** and label it in the subtitle. This is the single easiest way to publish
  a wrong number here.
- **Error bars = ±1 SE**, values in `eval_data.csv`; SE = `sqrt(p(1-p)/n)`.
- **Success rate axis: 0 to 0.45.** Do not truncate the y-axis — the differences are real but
  small, and a truncated axis overstates them.
- Colour: one accent for the scripted controller, one for learned policies, grey for the
  no-search walker. Keep "navigator better" bars visually distinct on slide 6.
- Full precision lives in the CSV; round to 3 decimals on slides, 4 in the appendix table.

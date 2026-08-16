Build a presentation deck for an engineering project review. Undergraduate robotics
project, technical audience (professors + engineers). Tone: precise, confident, not salesy.
This deck reports a NEGATIVE result honestly — do not spin it as a success story.

=== ABSOLUTE RULES — violating any of these makes the deck wrong ===

1. USE ONLY THE NUMBERS BELOW. Do not invent, round differently, interpolate, or "improve"
   any figure. If a number you want is not in this prompt, leave the space empty and write
   "[data not provided]" rather than inventing one.

2. NEVER put an n=256 number next to an n=1024 number in the same chart or table.
   The same controller measured 0.2109 and 0.2500 at n=256 but 0.1553 at n=1024. Sample size
   changes the terrain layout, not just precision. Every chart must fix one n and state it in
   the subtitle, e.g. "n = 1024 episodes".

3. DO NOT TRUNCATE the y-axis on any success-rate chart. Always start at 0. Max 0.45.
   The differences are real but small; a truncated axis overstates them and is dishonest.

4. Any slide using the no-rubble video or the 0.2266 figure MUST state that the debris was
   removed. Unlabelled it overstates the robot by 6.6 points of success.

5. Success rates are fractions 0-1. If you show percentages, convert consistently and label
   the axis "%". Never mix the two.

=== THE DATA (authoritative — this is the complete set) ===

--- Measurements at n = 1024 episodes (use these for all headline charts) ---
label                              success   fall_rate  SPL     falls_near_debris
Scripted seeker + walker            0.1553    0.5254    0.1073   0.1826
Learned navigator (init fixed)      0.0977    0.3809    0.0658   0.1270
Seeker + debris-finetuned walker    0.1592    0.4912    0.1153   0.1719
A/B arm: debris ON                  0.1602    0.5254    0.1167   0.1709
A/B arm: debris OFF                 0.2266    0.2754    0.1634   0.0000

--- Measurements at n = 256 episodes (older; use only on the training-curve slide) ---
label                     success
Walker alone, random cmd   0.1250
Scripted seeker, run A     0.2109
Scripted seeker, run B     0.2500
Navigator iteration 0      0.1289
Navigator iteration 100    0.1328
Navigator iteration 200    0.1289
Navigator iteration 299    0.1602

--- Significance tests (two-sided z on proportions) ---
Navigator vs seeker (n=1024):            delta -0.0576, p = 0.0001   SIGNIFICANT
Debris fine-tune vs control (n=1024):    delta +0.0039, p = 0.81     NOT significant
Debris OFF vs ON (n=1024):               delta +0.0664, p = 0.0001   SIGNIFICANT
Seeker run A vs run B (n=256):           delta +0.0391, p = 0.29     NOT significant
Fall rate, debris OFF vs ON:             delta -0.2500, p < 0.0001   SIGNIFICANT

--- Root cause 1: residual initialisation bug ---
Mean |command delta| as % of the +/-1.0 command clamp:
   Default init (as originally run):  15.01%
   Zero init (after fix):              0.86%
Consequence: navigator at iteration 0 scored 0.1289, statistically identical to the
random-command walker's 0.1250, instead of the scripted seeker's 0.2500.

--- Root cause 2: circular objective (reward hacking) ---
Reward term contributions at the final training iteration:
term                          corrected_run   original_run
track_lin_vel_xy_exp             0.4069          0.2320
track_ang_vel_z_exp              0.2853          0.2030
victim_found                     0.0565          0.0780
victim_approach                  0.0131          0.0140
victim_heading                   0.0510          0.0312
Mean total reward               11.70            6.80
Metrics/success_rate (command tracking, NOT rescue)   0.9272    0.2221
Key fact: tracking terms total 0.692 vs 0.121 for all victim terms = 5.7x imbalance.
Total reward nearly doubled while victim_found went DOWN.

--- Behaviour comparison, n = 1024 ---
metric              seeker    navigator
mean path length    5.96 m    3.36 m
mean time to find   6.06 s   10.15 s
mean episode length 12.17 s  14.76 s
fall rate            0.5254   0.3809

--- Phase 3 result: two-stage curriculum (n/a — training metrics, not eval) ---
                        single-stage   two-stage
victim_found reward        0.2406        0.1238
terrain_levels             0.2455        3.7186
Note: the FAILING run has the HIGHER victim reward. It farmed casualties on flat ground.

=== SYSTEM ARCHITECTURE (for the diagram slide) ===

Four layers, only ONE is trainable. Draw as a signal-flow diagram:

  thermal  ---------> scripted seeker ---> base command --+
                                                          +--> sum --> SHIELD --> walker
  thermal + LiDAR --> navigator (MLP) ---> delta ---------+                      (frozen)

layer               rate    trainable   job
Navigator (MLP)     10 Hz   YES         velocity-command correction from perception
Scripted seeker     50 Hz   no          base command from thermal centroid
LiDAR safety shield 50 Hz   no          vetoes forward motion within 1.0 m, +/-25 deg cone
Walker model_4197   50 Hz   FROZEN      tracks whatever command it is handed

Hardware/stack line for the title slide:
ANYmal-C quadruped - Isaac Sim 6.0.1 - Isaac Lab 3.0 - RSL-RL PPO - NVIDIA L40S

=== SLIDE STRUCTURE (12 slides + 4 appendix) ===

1. TITLE. "Rescue Robot: teaching a quadruped to search a disaster scene". Stack line above.

2. THE PROBLEM. Collapsed structure, quadruped with thermal camera + 288-ray LiDAR +
   emergency payload, 20 seconds to reach a casualty it cannot initially see.
   Success is defined as: reaching within 1.0 m of either of 2 casualties within 20 s.
   [Leave a full-bleed image placeholder labelled "hero frame - demo v6"]

3. ARCHITECTURE. The diagram above. Emphasise visually that only the navigator trains.

4. TWO DESIGN DECISIONS.
   (a) Navigator runs at 10 Hz not 50 Hz. With gamma=0.99 the credit horizon is ~100
       decisions = 2 seconds, against a 20-second episode, so credit for reaching a
       casualty never reaches the decision that turned toward it. Deciding 5x less often
       stretches the same horizon over 10 seconds.
   (b) The LiDAR shield sits OUTSIDE the network, after the command is summed, so no amount
       of training can remove it.

5. HEADLINE RESULT. Horizontal bar chart, n=1024, two bars, error bars = +/-1 SE
   (SE approx 0.011 both). Scripted seeker 0.1553 vs learned navigator 0.0977.
   Caption: delta -0.0576, p = 0.0001.
   This is a negative result. Present it plainly.

6. IT LEARNED THE WRONG THING. Use the behaviour comparison table above. The navigator
   falls less and survives longer because it barely moves. Colour-code "navigator better"
   vs "navigator worse" differently — the point is it wins on safety, loses on the mission.

7. ROOT CAUSE 1 - THE RESIDUAL WAS NEVER A RESIDUAL. Two-bar chart: 15.01% vs 0.86%.
   Explain: the design says the policy starts identical to the scripted controller so
   training can only improve on it. The code never implemented that — the output layer sat
   at default initialisation, injecting 15% of the full command range before any training.

8. ROOT CAUSE 2 - THE OBJECTIVE WAS CIRCULAR. Stacked bar of reward composition.
   The navigator emits the velocity command; the reward pays for tracking that command.
   So it learned to issue easy-to-follow orders and stop searching.
   Callout the number 0.9272 — it set its own exam and scored 93%.

9. WHAT THE RUBBLE COSTS. Paired bars, n=1024: debris ON 0.1602 vs OFF 0.2266 for success,
   and 0.5254 vs 0.2754 for fall rate. delta +0.0664, p = 0.0001.
   Also note the first attempt at this question was a confounded comparison that led to a
   500-iteration fine-tune which moved nothing (+0.0039, p = 0.81).

10. WHERE THE ROBOT STANDS. Three bars, n=1024: 0.1592 / 0.1553 / 0.0977.
    Plain-English caption: reaches a casualty about 16% of the time, falls in roughly half
    of attempts.

11. HOW THE NUMBERS WERE KEPT HONEST. Four points: every claim carries a significance test;
    sample size raised 256 -> 1024 when the question got finer; a falsifiable control for
    every claim (shield veto rate ~2-3% in both arms); negative results kept in the record.

12. NEXT STEPS, in priority order. (1) Train the walker on debris properly at ~1200
    iterations, not 500 — measured prize is 6.6 points of success. (2) Give the navigator a
    reward that is not its own command-tracking score — victim terms plus an explicit fall
    penalty. (3) Add memory; both controllers are stateless. (4) More seeds — every number
    here is one training run.

APPENDIX A1: full results table, grouped by n, with the n column visible.
APPENDIX A2: two-stage curriculum table (Phase 3 data above).
APPENDIX A3: thermal resolution — 64x64 made 94% of the observation vector camera data and
             collapsed episode length from 19-24 steps to 7.76. 16x16 fixed it.
APPENDIX A4: infrastructure — two cloud instances reclaimed within twelve hours, one
             mid-run; a checkpoint watchdog reduced a lost host to ~10 lost minutes.

=== DESIGN DIRECTION ===

Subject is field robotics and disaster response — instrument panels, sensor readouts,
telemetry. Draw from that world, not from generic startup-deck aesthetics.

- Palette: one restrained accent for the scripted controller, a second for learned policies,
  neutral grey for the no-search baseline. Semantic colour (good/bad) must be separate from
  the accent hue. Avoid the warm-cream-plus-terracotta and the purple-blue-gradient looks.
- Typography: pair a characterful display face with a clean body face; use tabular figures
  wherever digits align in columns.
- Charts: no truncated axes, no 3D, no gradients on bars, no decorative gridlines. Error bars
  wherever an SE is given. Label n in every chart subtitle.
- Dark and light themes should both read correctly if the format supports it.

=== DO NOT ===

- Do not invent any number, including axis ticks implying precision not in the data.
- Do not describe the project as a success or a failure. It is a measured negative result
  with two diagnosed causes.
- Do not write "improve the robot's vision" anywhere — higher camera resolution was measured
  as harmful in this project (see appendix A3).
- Do not claim any future performance figure. No "we expect 50%".
- Do not use emoji as section markers or bullets.
- Do not add a slide about commercial applications, market size, or deployment timelines.

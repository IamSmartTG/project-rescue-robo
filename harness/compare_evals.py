#!/usr/bin/env python3
"""Tabulate eval JSONs side by side, with the two-proportion test the comparison needs.

Exists because the Phase 5 numbers get quoted in three places -- the benchmark write-up, the
slide deck and the commit messages -- and hand-copying a success rate into three documents is
how a 0.1602 becomes a 0.1620 in the one that gets presented.

The significance test is here for a specific reason: at n=256 the standard error on a success
rate near 0.2 is about 0.025, so two runs can differ by 4 points and mean nothing. Any claim
that one policy beats another has to clear that bar explicitly rather than by eyeball.

Usage:
    python3 harness/compare_evals.py                    # every eval json in artifacts/benchmarks
    python3 harness/compare_evals.py --baseline "seeker (re-run)"
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

# Display name -> filename. Ordered as the story is told, not alphabetically: what the robot
# did with no search at all, then the hand-written controller, then the learned policy.
RUNS = [
    ("walker alone (random cmd)", "eval_rescue_debris_baseline.json"),
    ("scripted seeker (Aug 15)", "eval_scripted_seeker.json"),
    ("scripted seeker (Aug 16)", "eval_seeker_veto.json"),
    ("navigator iter 0", "eval_nav_0.json"),
    ("navigator iter 100", "eval_nav_100.json"),
    ("navigator iter 200", "eval_nav_200.json"),
    ("navigator iter 299", "eval_navigator.json"),
    # Corrected run: residual actually zero-initialised, so iteration 0 really is the seeker.
    # Evaluated at 1024 episodes because 256 cannot resolve the differences that matter here --
    # see the two scripted-seeker rows above, same config, 0.2109 vs 0.2500.
    ("seeker control (n=1024)", "eval_seeker_n1024.json"),
    ("zero-init nav i99 (n=1024)", "eval_zeroinit99_n1024.json"),
]


def se(p: float, n: int) -> float:
    """Standard error of a proportion. n is episodes, not envs -- they are equal here."""
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Unpooled z and a two-sided p, which is all a 256-episode comparison supports.

    Normal approximation rather than an exact test: with n=256 and p in the 0.1-0.3 range,
    np and n(1-p) are both well above 10, so the approximation is not the weak link. The weak
    link is that these are single runs with one seed.
    """
    d = p1 - p2
    s = math.sqrt(se(p1, n1) ** 2 + se(p2, n2) ** 2)
    if s == 0.0:
        return 0.0, 1.0
    z = d / s
    # Two-sided normal tail via erfc, so this needs no scipy.
    return z, math.erfc(abs(z) / math.sqrt(2.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(pathlib.Path(__file__).parent.parent / "artifacts" / "benchmarks"))
    ap.add_argument("--baseline", default="scripted seeker (Aug 16)", help="row every other row is tested against")
    args = ap.parse_args()

    base = pathlib.Path(args.dir)
    loaded: list[tuple[str, dict]] = []
    for name, fn in RUNS:
        p = base / fn
        if p.exists():
            loaded.append((name, json.loads(p.read_text())))
        else:
            print(f"[skip] {name}: {fn} not found")

    if not loaded:
        raise SystemExit("no eval JSONs found")

    cols = f"{'run':26} {'n':>4} {'succ':>7} {'±SE':>6} {'SPL':>6} {'fall':>6} {'debris':>7} {'veto':>7} {'path_m':>7} {'t_find':>7} {'ep_s':>6} {'terrain':>7}"
    print(cols)
    print("-" * len(cols))
    for name, d in loaded:
        n = d["episodes"]
        p = d["success_rate"]
        v = d.get("shield_veto_rate")
        print(
            f"{name:26} {n:4d} {p:7.4f} {se(p, n):6.4f} {d['spl_straight_line_proxy']:6.4f} "
            f"{d['fall_rate']:6.4f} {d['fall_rate_near_debris']:7.4f} "
            f"{('n/a' if v is None else f'{v:.4f}'):>7} {d['mean_path_length_m']:7.2f} "
            f"{d['mean_time_to_find_s']:7.3f} {d['mean_episode_length_s']:6.2f} {d['mean_terrain_level']:7.2f}"
        )

    ref = next((d for nm, d in loaded if nm == args.baseline), None)
    if ref is None:
        print(f"\n[warn] baseline '{args.baseline}' not present; skipping significance")
        return

    print(f"\nsuccess-rate vs '{args.baseline}' (two-sided z on proportions):")
    for name, d in loaded:
        if name == args.baseline:
            continue
        z, pv = two_proportion_z(d["success_rate"], d["episodes"], ref["success_rate"], ref["episodes"])
        delta = d["success_rate"] - ref["success_rate"]
        # 0.05 is a convention, not a discovery. Printed so the reader can disagree with it.
        verdict = "significant" if pv < 0.05 else "NOT significant"
        print(f"  {name:26} delta={delta:+.4f}  z={z:+5.2f}  p={pv:.4f}  {verdict}")


if __name__ == "__main__":
    main()

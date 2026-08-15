"""Success weighted by Path Length -- the standard target-finding metric.

Success rate alone cannot tell "found the casualty by searching" from "wandered into one".
SPL can: it pays an episode by how close the path walked came to the shortest path that
would have worked, and pays nothing at all for a failure.

    SPL = (1/N) * sum_i  S_i * l_i / max(p_i, l_i)

    S_i  1 if the episode found a casualty, else 0
    l_i  shortest path that would have counted as success
    p_i  path the robot actually walked before first contact

From Anderson et al. 2018, "On Evaluation of Embodied Navigation Agents" (arXiv:1807.06757),
the metric the Habitat / HM3D / RoboTHOR ObjectNav challenges report.

.. important::
    ``l_i`` here is **straight-line, not geodesic**. The definition calls for the geodesic
    shortest path, which on a map with walls is materially shorter than the crow-flies
    distance. ``ROUGH_TERRAINS_CFG`` has no walls -- it is open ground with relief, and the
    casualties spawn 1.6-3.6 m away inside the same 8x8 m patch -- so straight-line is a
    close proxy. It is still a proxy. Report this number as "SPL (straight-line proxy)" and
    never as plain SPL, or the comparison to a published ObjectNav figure is dishonest: this
    approximation can only ever flatter the agent, since a true geodesic path is >= the
    straight line and a larger ``l_i`` raises the ratio.

Deliberately pure Python on plain floats, with no torch. The reduction runs once over a few
hundred finished episodes, so vectorising it would buy nothing, and staying dependency-free
is what lets the self-check at the bottom run anywhere -- including a laptop with no torch
installed, which is where this gets edited.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["spl", "shortest_path_length"]


def shortest_path_length(spawn_distance: float, found_radius: float) -> float:
    """Shortest path that would have counted as a success, in metres.

    Success is defined as getting *within* ``found_radius`` of a casualty, not as standing on
    it, so the robot only has to close ``spawn_distance - found_radius``. Using the raw
    spawn distance instead would overstate the optimal path by the found radius and inflate
    every SPL score.

    Clamped at zero for the case where a casualty spawns already inside the found radius.
    """
    return max(spawn_distance - found_radius, 0.0)


def spl(
    success: Sequence[bool],
    path_length: Sequence[float],
    shortest: Sequence[float],
) -> float:
    """Mean SPL over a batch of finished episodes.

    Args:
        success: Whether each episode found a casualty.
        path_length: Distance each robot actually walked before first contact [m]. For a
            failed episode this is ignored, so it does not matter what is passed.
        shortest: Shortest successful path per episode [m], from
            :func:`shortest_path_length`.

    Returns:
        Mean SPL in ``[0, 1]``. Zero for an empty batch.
    """
    if not (len(success) == len(path_length) == len(shortest)):
        raise ValueError(
            f"ragged inputs: {len(success)} success, {len(path_length)} path, {len(shortest)} shortest"
        )
    if not success:
        return 0.0

    total = 0.0
    for hit, walked, optimal in zip(success, path_length, shortest):
        if not hit:
            continue
        if optimal <= 0.0:
            # Casualty spawned inside found_radius: the optimal path is zero-length, so any
            # success is by definition optimal. The ratio would be 0/0 otherwise.
            total += 1.0
            continue
        # max() in the denominator is what bounds this at 1.0. A robot cannot score above
        # perfect by appearing to walk less than the straight line -- which does happen, from
        # sampling the path as a polyline at 50 Hz while the casualty is a 1.75 m box whose
        # near face is closer than its centre.
        total += optimal / max(walked, optimal)
    return total / len(success)


def _self_check() -> None:
    """Assert the properties that matter, since a wrong SPL is a wrong headline number."""
    # Optimal path walked exactly -> perfect score.
    assert spl([True], [2.0], [2.0]) == 1.0

    # Twice the optimal path -> half credit.
    assert spl([True], [4.0], [2.0]) == 0.5

    # A failure contributes nothing, but still counts in the denominator: one perfect run and
    # one failure is 0.5, NOT 1.0. This is the property that makes SPL a joint measure of
    # success and efficiency rather than efficiency over successes only.
    assert spl([True, False], [2.0, 99.0], [2.0, 2.0]) == 0.5

    # Walking less than the straight line is clamped to 1.0, never above it.
    assert spl([True], [1.0], [2.0]) == 1.0

    # Zero-length optimal path (casualty spawned inside found_radius) is a success, not a NaN.
    assert spl([True], [0.0], [0.0]) == 1.0
    assert spl([False], [0.0], [0.0]) == 0.0

    # All failures -> 0.0, and an empty batch does not divide by zero.
    assert spl([False, False], [1.0, 1.0], [2.0, 2.0]) == 0.0
    assert spl([], [], []) == 0.0

    # found_radius is subtracted, and the subtraction clamps rather than going negative.
    assert shortest_path_length(3.6, 1.0) == 2.6
    assert shortest_path_length(0.4, 1.0) == 0.0

    # Ragged input is a bug in the caller, not something to average over silently.
    try:
        spl([True], [1.0, 2.0], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("ragged inputs must raise")

    print("spl.py self-check passed")


if __name__ == "__main__":
    _self_check()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pseudo-thermal (IR) synthesis from semantic segmentation.

Isaac Sim has no native IR / thermal sensor. This module fakes one: a normal RTX camera
renders a ``semantic_segmentation`` buffer, each semantic class is assigned a nominal
surface temperature, and the segmentation image is converted into a temperature image
via a lookup table.

The camera is configured with ``colorize_semantic_segmentation=True`` and an explicit
:data:`SEMANTIC_SEGMENTATION_MAPPING`, so the renderer writes a deterministic RGBA color
per class (shape ``(N, H, W, 4)``, dtype ``uint8`` -- verified in
``isaaclab_physx/renderers/isaac_rtx_renderer.py::supported_output_types``). The LUT is
therefore keyed on the exact RGB triplet, which avoids having to parse the per-env
``idToLabels`` info dict at runtime.
"""

from __future__ import annotations

import torch

##
# Class -> nominal temperature [deg C]
##

AMBIENT_TEMPERATURE_C: float = 12.0
"""Temperature assigned to any pixel whose color is not in the LUT (unlabelled / background)."""

CLASS_TEMPERATURE_C: dict[str, float] = {
    "victim": 36.5,  # human body surface temperature -- the search target
    "robot": 30.0,  # motors / battery, warm but cooler than a human
    "terrain": 15.0,  # ground
}
"""Nominal surface temperature (deg C) per semantic class.

.. note::
    ``TerrainImporterCfg`` has no spawner, so the generated terrain currently carries no
    ``class`` semantic tag and its pixels fall back to :data:`AMBIENT_TEMPERATURE_C`. The
    ``terrain`` entry is kept so a tagged ground mesh works without a code change.
"""

TEMPERATURE_RANGE_C: tuple[float, float] = (10.0, 40.0)
"""(min, max) temperature used to normalize the thermal image into [0, 1]."""

##
# Class -> segmentation color (RGBA, uint8)
##

CLASS_SEG_COLOR: dict[str, tuple[int, int, int, int]] = {
    "victim": (255, 0, 0, 255),
    "robot": (0, 0, 255, 255),
    "terrain": (128, 128, 128, 255),
}
"""Deterministic RGBA color the renderer must paint each class with."""

SEMANTIC_SEGMENTATION_MAPPING: dict[str, tuple[int, int, int, int]] = {
    f"class:{name}": color for name, color in CLASS_SEG_COLOR.items()
}
"""Value for ``IsaacRtxRendererCfg.semantic_segmentation_mapping``."""

SEMANTIC_FILTER: str = "class:*"
"""Value for ``IsaacRtxRendererCfg.semantic_filter`` -- only the ``class`` semantic type."""


##
# LUT construction / lookup
##

_LUT_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def build_lut(device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (and cache) the color -> temperature lookup tables.

    Args:
        device: Device to allocate the tables on.

    Returns:
        A tuple ``(colors, temperatures)`` where ``colors`` has shape ``(K, 3)`` (dtype
        ``uint8``, RGB only) and ``temperatures`` has shape ``(K,)`` (dtype ``float32``,
        deg C). ``K`` is the number of known classes.
    """
    key = str(device)
    if key not in _LUT_CACHE:
        names = list(CLASS_SEG_COLOR.keys())
        colors = torch.tensor([CLASS_SEG_COLOR[n][:3] for n in names], dtype=torch.uint8, device=device)
        temps = torch.tensor([CLASS_TEMPERATURE_C[n] for n in names], dtype=torch.float32, device=device)
        _LUT_CACHE[key] = (colors, temps)
    return _LUT_CACHE[key]


def segmentation_to_temperature(seg_rgba: torch.Tensor) -> torch.Tensor:
    """Convert a colorized semantic segmentation image into a temperature image.

    Args:
        seg_rgba: Colorized segmentation buffer. Shape ``(N, H, W, 4)`` (or ``(N, H, W, 3)``),
            dtype ``uint8``.

    Returns:
        Temperature image in deg C. Shape ``(N, H, W)``, dtype ``float32``.
    """
    colors, temps = build_lut(seg_rgba.device)
    rgb = seg_rgba[..., :3]
    # (N, H, W, 1, 3) == (K, 3) -> (N, H, W, K)
    match = (rgb.unsqueeze(-2) == colors).all(dim=-1)
    # index of the matched class; 0 where nothing matched (masked out below)
    idx = match.to(torch.uint8).argmax(dim=-1)
    known = match.any(dim=-1)
    out = temps[idx.long()]
    return torch.where(known, out, torch.full_like(out, AMBIENT_TEMPERATURE_C))


def normalize_temperature(temperature_c: torch.Tensor) -> torch.Tensor:
    """Scale a temperature image (deg C) into ``[0, 1]`` using :data:`TEMPERATURE_RANGE_C`."""
    lo, hi = TEMPERATURE_RANGE_C
    return ((temperature_c - lo) / (hi - lo)).clamp(0.0, 1.0)


##
# Colormap helper (visualization / debugging only -- not used in the observation path)
##

# Ironbow-ish control points: black -> purple -> red -> orange -> yellow -> white.
_COLORMAP_STOPS: torch.Tensor = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.28, 0.0, 0.45],
        [0.72, 0.0, 0.25],
        [0.98, 0.45, 0.0],
        [1.0, 0.85, 0.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


def temperature_to_colormap(temperature_c: torch.Tensor) -> torch.Tensor:
    """Map a temperature image to an ironbow RGB image for visualization.

    Args:
        temperature_c: Temperature image in deg C. Shape ``(..., H, W)``.

    Returns:
        RGB image in ``[0, 1]``. Shape ``(..., H, W, 3)``, dtype ``float32``.
    """
    stops = _COLORMAP_STOPS.to(temperature_c.device)
    t = normalize_temperature(temperature_c) * (stops.shape[0] - 1)
    lo = t.floor().clamp(0, stops.shape[0] - 2).long()
    frac = (t - lo).unsqueeze(-1)
    return stops[lo] * (1.0 - frac) + stops[lo + 1] * frac

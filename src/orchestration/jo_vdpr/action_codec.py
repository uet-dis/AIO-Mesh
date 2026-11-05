"""Encode the placement/routing pair as one Discrete(N^2) action."""

from __future__ import annotations

import operator
from typing import Any, Tuple

import numpy as np


def encode_pair_action(v_place: int, v_route: int, num_nodes: int) -> int:
    """Map a node pair to the flattened categorical action index."""
    n = operator.index(num_nodes)
    place = operator.index(v_place)
    route = operator.index(v_route)
    if n <= 0:
        raise ValueError("num_nodes must be positive")
    if not 0 <= place < n or not 0 <= route < n:
        raise ValueError(f"placement and route must be in [0, {n})")
    return place * n + route


def decode_pair_action(action: Any, num_nodes: int) -> Tuple[int, int]:
    """Decode a flat action index; retain pair input for heuristic callers."""
    n = operator.index(num_nodes)
    if n <= 0:
        raise ValueError("num_nodes must be positive")

    values = np.asarray(action)
    if values.ndim == 0 or values.size == 1:
        flat = operator.index(values.reshape(-1)[0])
        if not 0 <= flat < n * n:
            raise ValueError(f"flat action must be in [0, {n * n})")
        return flat // n, flat % n

    if values.size != 2:
        raise ValueError("pair action must contain exactly placement and route")
    place, route = (operator.index(value) for value in values.reshape(-1))
    if not 0 <= place < n or not 0 <= route < n:
        raise ValueError(f"placement and route must be in [0, {n})")
    return place, route

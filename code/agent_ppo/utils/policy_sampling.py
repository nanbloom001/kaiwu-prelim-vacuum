#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Probability sanitization helpers for action sampling.
"""

from __future__ import annotations

import math
import random


def _normalize_legal_mask(legal_action) -> list[float]:
    mask = [1.0 if float(value) > 0.5 else 0.0 for value in list(legal_action or [])]
    return mask


def uniform_over_legal(legal_action) -> list[float]:
    legal_mask = _normalize_legal_mask(legal_action)
    if not legal_mask:
        return []
    legal_count = sum(1 for value in legal_mask if value > 0.5)
    if legal_count <= 0:
        uniform = 1.0 / len(legal_mask)
        return [uniform for _ in legal_mask]
    return [(1.0 / legal_count) if value > 0.5 else 0.0 for value in legal_mask]


def sanitize_policy_probs(probs, legal_action) -> tuple[list[float], bool]:
    values = [float(value) for value in list(probs or [])]
    if not values:
        return uniform_over_legal(legal_action), True

    legal_mask = _normalize_legal_mask(legal_action)
    if len(legal_mask) != len(values):
        legal_mask = [1.0] * len(values)

    cleaned = []
    for value, legal in zip(values, legal_mask):
        if not math.isfinite(value):
            value = 0.0
        value = max(0.0, min(1.0, value))
        cleaned.append(value * legal)

    total = sum(cleaned)
    if total <= 1e-8:
        return uniform_over_legal(legal_mask), True

    normalized = [value / total for value in cleaned]
    total = sum(normalized)
    if total <= 1e-8:
        return uniform_over_legal(legal_mask), True
    normalized = [value / total for value in normalized]
    if any((not math.isfinite(value)) or value < -1e-8 or value > 1.0 + 1e-8 for value in normalized):
        return uniform_over_legal(legal_mask), True

    return normalized, normalized != values


def safe_sample_action(probs, legal_action, use_max: bool = False, rng_seed: int | None = None) -> dict[str, object]:
    safe_probs, used_fallback = sanitize_policy_probs(probs, legal_action)
    if not safe_probs:
        return {"action": 0, "probs": [1.0], "used_fallback": True}
    if use_max:
        action = max(range(len(safe_probs)), key=lambda idx: safe_probs[idx])
    else:
        rng = random.Random(rng_seed)
        draw = rng.random()
        cumulative = 0.0
        action = len(safe_probs) - 1
        for idx, value in enumerate(safe_probs):
            cumulative += value
            if draw <= cumulative:
                action = idx
                break
    return {
        "action": int(action),
        "probs": safe_probs,
        "used_fallback": bool(used_fallback),
    }

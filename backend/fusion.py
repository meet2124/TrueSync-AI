"""
backend/fusion.py
=================
TrueSync AI — Production V1 Weighted Score Fusion.

Combines rPPG, acoustic, and sync sub-scores into a single overall_trust (0–100).

Weights are MODULE-LEVEL CONSTANTS — inspectable and tunable by judges/teammates.
If a sub-score is None (module still calibrating), it is excluded and remaining
weights are re-normalised. Status reflects calibration state.

overall_trust = 100 × Σ(w_i × score_i) / Σ(w_i)   for available scores only

Signal state distinction
------------------------
AVAILABLE     : score is a float in [0, 1] — module has sufficient data.
LOW_CONFIDENCE: overall_trust < THRESHOLD_LOW_CONFIDENCE — signal present but weak.
UNAVAILABLE   : score is None — module has not yet accumulated minimum buffer,
                or encountered an unrecoverable error. Never treated as 0 or 0.5.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

logger = logging.getLogger("truesync.fusion")

# ── Tunable fusion weights ────────────────────────────────────────────────────
W_RPPG: float = 0.40      # rPPG liveness carries most weight (direct biological signal)
W_ACOUSTIC: float = 0.35  # Acoustic phase/flatness anti-spoof
W_SYNC: float = 0.25      # Viseme-phoneme sync (deepfake indicator)

# ── Decision thresholds (for status classification) ───────────────────────────
THRESHOLD_LOW_CONFIDENCE: float = 50.0    # below this → "low_confidence"
THRESHOLD_NOMINAL: float = 75.0           # above this → "nominal"


def _clamp_score(score: Optional[float]) -> Optional[float]:
    """
    Clamp a sub-score to [0, 1] and reject NaN/Inf.
    Returns None if the value is not a valid finite number.
    """
    if score is None:
        return None
    if not math.isfinite(score):
        logger.warning("fusion: received non-finite sub-score (%.6g) — treated as unavailable", score)
        return None
    return max(0.0, min(1.0, score))


def fuse(
    rppg_confidence: Optional[float],
    acoustic_trust: Optional[float],
    sync_score: Optional[float],
) -> Dict:
    """
    Fuse available sub-scores into overall_trust.

    Parameters
    ----------
    rppg_confidence : float | None  — rPPG liveness confidence [0, 1]
    acoustic_trust  : float | None  — acoustic anti-spoof score [0, 1]
    sync_score      : float | None  — viseme-phoneme sync [0, 1]

    Returns
    -------
    dict with keys: overall_trust (float | None), status (str)
    """
    candidates = [
        (_clamp_score(rppg_confidence), W_RPPG, "rppg"),
        (_clamp_score(acoustic_trust), W_ACOUSTIC, "acoustic"),
        (_clamp_score(sync_score), W_SYNC, "sync"),
    ]

    available = [(score, w, name) for score, w, name in candidates if score is not None]

    if not available:
        return {
            "overall_trust": None,
            "status": "insufficient_data",
        }

    # Weighted average over available sub-scores only
    total_weight = sum(w for _, w, _ in available)
    weighted_sum = sum(score * w for score, w, _ in available)
    overall_01 = weighted_sum / total_weight  # 0–1
    # Final clamp: guard against any floating-point edge cases
    overall_100 = round(max(0.0, min(100.0, overall_01 * 100.0)), 2)

    # Status classification
    n_available = len(available)
    n_total = len(candidates)

    if n_available < n_total:
        status = "calibrating"
    elif overall_100 < THRESHOLD_LOW_CONFIDENCE:
        status = "low_confidence"
    elif overall_100 >= THRESHOLD_NOMINAL:
        status = "nominal"
    else:
        status = "low_confidence"

    logger.debug(
        "Fusion: rppg=%s acoustic=%s sync=%s → overall=%.2f status=%s",
        f"{rppg_confidence:.3f}" if rppg_confidence is not None else "None",
        f"{acoustic_trust:.3f}" if acoustic_trust is not None else "None",
        f"{sync_score:.3f}" if sync_score is not None else "None",
        overall_100,
        status,
    )

    return {
        "overall_trust": overall_100,
        "status": status,
    }

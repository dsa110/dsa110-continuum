"""Lightweight flux scale validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FluxScaleCheckResult:
    """Result from a quick MODEL_DATA vs CORRECTED_DATA flux scale check."""

    status: str
    ratio: float
    factor: float
    model_median: float
    corrected_median: float
    n_samples: int
    n_flagged: int
    message: str


def check_model_corrected_ratio(
    ms_path: str | Path,
    *,
    sample_rows: int = 500,
    channel_fraction: float = 0.5,
    ratio_warn: float = 1.5,
    ratio_fail: float = 5.0,
) -> FluxScaleCheckResult:
    """Compare MODEL_DATA vs CORRECTED_DATA amplitude to flag gross flux mismatches.

    This is a lightweight, MS-level check intended to catch missing applycal or
    large flux scale gaps early. It samples a subset of rows and central channels
    to avoid heavy I/O.
    """
    import casacore.tables as tb
    import numpy as np

    ms_path = str(ms_path)
    with tb.table(ms_path, readonly=True) as table:
        cols = set(table.colnames())
        if "MODEL_DATA" not in cols or "CORRECTED_DATA" not in cols:
            return FluxScaleCheckResult(
                status="missing",
                ratio=float("nan"),
                factor=float("inf"),
                model_median=float("nan"),
                corrected_median=float("nan"),
                n_samples=0,
                n_flagged=0,
                message="MODEL_DATA or CORRECTED_DATA missing; cannot validate flux scale.",
            )

        nrows = table.nrows()
        if nrows == 0:
            return FluxScaleCheckResult(
                status="missing",
                ratio=float("nan"),
                factor=float("inf"),
                model_median=float("nan"),
                corrected_median=float("nan"),
                n_samples=0,
                n_flagged=0,
                message="MS has zero rows; cannot validate flux scale.",
            )

        nrow = min(sample_rows, nrows)
        startrow = max(0, (nrows - nrow) // 2)

        model = table.getcol("MODEL_DATA", startrow=startrow, nrow=nrow)
        corrected = table.getcol("CORRECTED_DATA", startrow=startrow, nrow=nrow)
        flags = table.getcol("FLAG", startrow=startrow, nrow=nrow) if "FLAG" in cols else None

    if model.ndim != 3 or corrected.ndim != 3:
        return FluxScaleCheckResult(
            status="missing",
            ratio=float("nan"),
            factor=float("inf"),
            model_median=float("nan"),
            corrected_median=float("nan"),
            n_samples=0,
            n_flagged=0,
            message="Unexpected DATA shape; expected (npol, nchan, nrow).",
        )

    nchan = model.shape[1]
    if nchan == 0:
        return FluxScaleCheckResult(
            status="missing",
            ratio=float("nan"),
            factor=float("inf"),
            model_median=float("nan"),
            corrected_median=float("nan"),
            n_samples=0,
            n_flagged=0,
            message="MS has zero channels; cannot validate flux scale.",
        )

    keep_fraction = max(0.1, min(1.0, channel_fraction))
    chan_trim = int((1.0 - keep_fraction) * nchan / 2)
    chan_slice = slice(chan_trim, nchan - chan_trim)

    model_amp = np.abs(model[:, chan_slice, :])
    corrected_amp = np.abs(corrected[:, chan_slice, :])

    if flags is not None:
        flag_slice = flags[:, chan_slice, :]
        valid = ~flag_slice
        n_flagged = int(flag_slice.sum())
        model_vals = model_amp[valid]
        corrected_vals = corrected_amp[valid]
    else:
        n_flagged = 0
        model_vals = model_amp.ravel()
        corrected_vals = corrected_amp.ravel()

    if model_vals.size == 0 or corrected_vals.size == 0:
        return FluxScaleCheckResult(
            status="missing",
            ratio=float("nan"),
            factor=float("inf"),
            model_median=float("nan"),
            corrected_median=float("nan"),
            n_samples=0,
            n_flagged=n_flagged,
            message="No unflagged samples available; cannot validate flux scale.",
        )

    model_median = float(np.nanmedian(model_vals))
    corrected_median = float(np.nanmedian(corrected_vals))
    if corrected_median <= 0:
        ratio = float("nan")
        factor = float("inf")
    else:
        ratio = model_median / corrected_median
        factor = max(ratio, 1.0 / ratio) if ratio > 0 else float("inf")

    if not np.isfinite(factor):
        status = "fail"
        message = "Flux scale check failed: non-finite ratio."
    elif factor >= ratio_fail:
        status = "fail"
        message = f"Flux scale factor {factor:.2f}x exceeds fail threshold ({ratio_fail:.2f}x)."
    elif factor >= ratio_warn:
        status = "warn"
        message = f"Flux scale factor {factor:.2f}x exceeds warn threshold ({ratio_warn:.2f}x)."
    else:
        status = "ok"
        message = f"Flux scale factor {factor:.2f}x within threshold."

    return FluxScaleCheckResult(
        status=status,
        ratio=ratio,
        factor=factor,
        model_median=model_median,
        corrected_median=corrected_median,
        n_samples=int(model_vals.size),
        n_flagged=n_flagged,
        message=message,
    )

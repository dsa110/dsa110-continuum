"""Light curve stacking and variability analysis for DSA-110 continuum pipeline.

Modules
-------
stacker
    Stack per-epoch forced-photometry CSVs into a cross-epoch Parquet light
    curve table, with source cross-matching and stable source IDs.
metrics
    Compute per-source Mooley et al. (2016) variability metrics:
    modulation index *m*, flux variability significance *Vs*, and reduced
    chi-squared *η*.

Typical usage::

    from dsa110_continuum.lightcurves.stacker import stack_csvs, parse_epoch_utc
    from dsa110_continuum.lightcurves.metrics import (
        compute_metrics,
        flag_candidates,
        VariabilityMetrics,
    )

    df = stack_csvs(csv_paths)
    metrics = compute_metrics(df)
    candidates = flag_candidates(metrics)
"""

from dsa110_continuum.lightcurves.metrics import (
    VariabilityMetrics,
    compute_metrics,
    flag_candidates,
)
from dsa110_continuum.lightcurves.stacker import (
    assign_source_ids,
    parse_epoch_utc,
    stack_csvs,
)

__all__ = [
    "stack_csvs",
    "assign_source_ids",
    "parse_epoch_utc",
    "compute_metrics",
    "flag_candidates",
    "VariabilityMetrics",
]

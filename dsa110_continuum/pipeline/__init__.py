"""Per-epoch orchestration layer for the DSA-110 continuum pipeline.

Provides :class:`EpochOrchestrator` — a reusable, test-friendly alternative
to the monolithic ``scripts/batch_pipeline.py``.  The orchestrator:

* Accepts a list of FITS tile paths for one epoch
* Mosaics them using :func:`~dsa110_continuum.mosaic.builder.fast_reproject_and_coadd`
* Evaluates all three QA gates via :class:`~dsa110_continuum.qa.composite.CompositeQA`
* Persists per-epoch results to a SQLite database (path from ``PathConfig``)
* Returns a structured :class:`EpochRunResult` with accept / warn / reject decision

Typical usage::

    from dsa110_continuum.pipeline import EpochOrchestrator

    orch = EpochOrchestrator()
    result = orch.run_epoch("2026-01-25T22:00:00", tile_paths=[...])
    if result.accepted:
        downstream_process(result.mosaic_path)
"""

from dsa110_continuum.pipeline.epoch_orchestrator import (
    EpochDecision,
    EpochOrchestrator,
    EpochRunResult,
)

__all__ = ["EpochOrchestrator", "EpochRunResult", "EpochDecision"]

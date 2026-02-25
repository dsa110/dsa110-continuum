"""
Source class for DSA-110 photometry analysis.

Adopted from VAST Tools pattern for representing sources with measurements
across multiple epochs. Provides clean interface for ESE candidate analysis.

Reference: archive/references/vast-tools/vasttools/source.py
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

try:
    from dsa110_contimg.core.catalog.multiwavelength import (
        check_all_services,
        check_atnf,
        check_first,
        check_gaia,
        check_nvss,
        check_pulsarscraper,
        check_simbad,
    )
    from dsa110_contimg.core.photometry.variability import (
        calculate_eta_metric,
        calculate_m_metric,
        calculate_vs_metric,
    )
except ImportError:
    # Fallback
    def check_all_services(*args, **kwargs):
        return {}

    def calculate_eta_metric(*args, **kwargs):
        return 0.0

    def calculate_vs_metric(*args, **kwargs):
        return 0.0

    def calculate_m_metric(*args, **kwargs):
        return 0.0


logger = logging.getLogger(__name__)


class SourceError(Exception):
    """Error raised by Source operations."""


class Source:
    """Represents a single source with measurements across multiple epochs.

        Provides a clean interface for ESE candidate analysis, including:
        - Light curve plotting
        - Variability metric calculations
        - Measurement access and filtering

    Attributes
    ----------
    source_id : str
        Source identifier (e.g., 'NVSS J123456+420312')
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    name : str or None
        Optional display name for the source
        products_db :
        Path to products database
    measurements : DataFrame
        DataFrame containing all photometry measurements
    """

    def __init__(
        self,
        source_id: str,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        name: str | None = None,
        products_db: Path | None = None,
    ) -> None:
        """Initialize Source object.

        Parameters
        ----------
        source_id : str
            Source identifier (e.g., 'NVSS J123456+420312')
        ra_deg : float, optional
            Right ascension in degrees (optional, will be loaded from DB)
        dec_deg : float, optional
            Declination in degrees (optional, will be loaded from DB)
        name : str, optional
            Optional display name (defaults to source_id)
        products_db : str
            Path to products database (required for loading measurements)

        Raises
        ------
            SourceError
            If products_db is None or measurements cannot be loaded
        """
        self.source_id = source_id
        self.name = name or source_id
        self.products_db = Path(products_db) if products_db else None

        # Load measurements from database
        if self.products_db:
            self.measurements = self._load_measurements()
            # Get coordinates from first measurement if not provided
            if ra_deg is None or dec_deg is None:
                if len(self.measurements) > 0:
                    self.ra_deg = float(self.measurements.iloc[0]["ra_deg"])
                    self.dec_deg = float(self.measurements.iloc[0]["dec_deg"])
                else:
                    raise SourceError(
                        f"No measurements found for source {source_id} and coordinates not provided"
                    )
            else:
                self.ra_deg = ra_deg
                self.dec_deg = dec_deg
        else:
            if ra_deg is None or dec_deg is None:
                raise SourceError(
                    "Either products_db must be provided or both ra_deg and "
                    "dec_deg must be specified"
                )
            self.ra_deg = ra_deg
            self.dec_deg = dec_deg
            self.measurements = pd.DataFrame()

        logger.debug(f"Created Source instance for {self.source_id}")

    def _load_measurements(self) -> pd.DataFrame:
        """Load photometry measurements from products database.

        Tries to load from photometry_timeseries table first, falls back to
        photometry table if needed.

        """
        if not self.products_db or not self.products_db.exists():
            raise SourceError(f"Products database not found: {self.products_db}")

        conn = sqlite3.connect(str(self.products_db), timeout=30.0)
        conn.row_factory = sqlite3.Row

        try:
            # Check what tables exist
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "photometry" in tables:
                # Check if source_id column exists
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(photometry)").fetchall()
                }

                if "source_id" in columns:
                    query = """
                        SELECT
                            ra_deg, dec_deg, nvss_flux_mjy,
                            peak_jyb, peak_err_jyb, measured_at, mjd, image_path
                        FROM photometry
                        WHERE source_id = ? OR source_id LIKE ?
                        ORDER BY measured_at ASC
                    """
                    df = pd.read_sql_query(
                        query, conn, params=(self.source_id, f"%{self.source_id}%")
                    )

                    # Calculate MJD if missing
                    if "mjd" not in df.columns or df["mjd"].isna().all():
                        if "measured_at" in df.columns:
                            df["mjd"] = pd.to_datetime(
                                df["measured_at"], unit="s", errors="coerce"
                            ).apply(lambda x: Time(x).mjd if pd.notna(x) else None)

                    # Add normalized flux columns (will be NaN if not available)
                    df["normalized_flux_jy"] = np.nan
                    df["normalized_flux_err_jy"] = np.nan
                else:
                    # No source_id column, can't query
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()

            conn.close()

            if df.empty:
                logger.warning(f"No measurements found for source {self.source_id}")
                return pd.DataFrame()

            # Ensure required columns exist
            required_cols = ["mjd", "peak_jyb", "peak_err_jyb", "image_path"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in measurements: {missing_cols}")

            # Ensure normalized flux columns exist (populate with None/NaN if missing)
            if "normalized_flux_jy" not in df.columns:
                df["normalized_flux_jy"] = np.nan
            if "normalized_flux_err_jy" not in df.columns:
                df["normalized_flux_err_jy"] = np.nan

            # Convert measured_at to datetime if present
            if "measured_at" in df.columns:
                df["measured_at"] = pd.to_datetime(df["measured_at"], unit="s", errors="coerce")

            return df

        except Exception as e:
            conn.close()
            raise SourceError(f"Failed to load measurements for {self.source_id}: {e}") from e

    @property
    def coord(self) -> SkyCoord:
        """Source coordinates as SkyCoord object."""
        return SkyCoord(self.ra_deg, self.dec_deg, unit=(u.deg, u.deg))

    @property
    def n_epochs(self) -> int:
        """Number of epochs with measurements."""
        return len(self.measurements)

    @property
    def detections(self) -> int:
        """Number of detections (SNR > 5 or flux > 3*error).

        Uses peak_jyb and peak_err_jyb if available, otherwise estimates
        from normalized_flux_jy and normalized_flux_err_jy.

        """
        if self.measurements.empty:
            return 0

        # Try to use SNR column if available
        if "snr" in self.measurements.columns:
            return (self.measurements["snr"] > 5).sum()

        # Estimate from flux and error
        flux_col = (
            "normalized_flux_jy"
            if "normalized_flux_jy" in self.measurements.columns
            else "peak_jyb"
        )
        err_col = (
            "normalized_flux_err_jy"
            if "normalized_flux_err_jy" in self.measurements.columns
            else "peak_err_jyb"
        )

        if flux_col in self.measurements.columns and err_col in self.measurements.columns:
            flux = self.measurements[flux_col]
            err = self.measurements[err_col]
            # Detection if flux > 3*error and error is finite
            mask = (flux > 3 * err) & np.isfinite(err) & (err > 0)
            return mask.sum()

        # Fallback: assume all are detections if we can't determine
        return len(self.measurements)

    def calc_variability_metrics(self) -> dict[str, Any]:
        """Calculate variability metrics for the source."""
        if len(self.measurements) < 2:
            return {
                "v": 0.0,
                "eta": 0.0,
                "vs_mean": None,
                "m_mean": None,
                "n_epochs": len(self.measurements),
            }

        # Use normalized flux if available, otherwise peak flux
        flux_col = (
            "normalized_flux_jy"
            if "normalized_flux_jy" in self.measurements.columns
            else "peak_jyb"
        )
        err_col = (
            "normalized_flux_err_jy"
            if "normalized_flux_err_jy" in self.measurements.columns
            else "peak_err_jyb"
        )

        if flux_col not in self.measurements.columns:
            raise SourceError(f"Flux column {flux_col} not found in measurements")

        flux = self.measurements[flux_col].values
        flux_err = (
            self.measurements[err_col].values if err_col in self.measurements.columns else None
        )

        # Filter out NaN/inf values
        valid_mask = np.isfinite(flux)
        if flux_err is not None:
            valid_mask &= np.isfinite(flux_err) & (flux_err > 0)

        if valid_mask.sum() < 2:
            return {
                "v": 0.0,
                "eta": 0.0,
                "vs_mean": None,
                "m_mean": None,
                "n_epochs": len(self.measurements),
            }

        flux_valid = flux[valid_mask]
        flux_err_valid = flux_err[valid_mask] if flux_err is not None else None

        # V metric (coefficient of variation)
        v = float(np.std(flux_valid) / np.mean(flux_valid)) if np.mean(flux_valid) > 0 else 0.0

        # η metric (weighted variance)
        if flux_err_valid is not None and len(flux_valid) >= 2:
            df_for_eta = pd.DataFrame({flux_col: flux_valid, err_col: flux_err_valid})
            try:
                eta = calculate_eta_metric(df_for_eta, flux_col=flux_col, err_col=err_col)
            except Exception as e:
                logger.warning(f"Failed to calculate η metric: {e}")
                # Use None instead of 0.0 to avoid falsely labeling unstable sources as stable
                eta = None
        else:
            eta = 0.0

        # Two-epoch metrics
        vs_metrics = []
        m_metrics = []
        if len(flux_valid) >= 2:
            for i in range(len(flux_valid) - 1):
                if flux_err_valid is not None:
                    vs = calculate_vs_metric(
                        flux_valid[i],
                        flux_valid[i + 1],
                        flux_err_valid[i],
                        flux_err_valid[i + 1],
                    )
                    vs_metrics.append(vs)
                m = calculate_m_metric(flux_valid[i], flux_valid[i + 1])
                m_metrics.append(m)

        return {
            "v": v,
            "eta": float(eta),
            "vs_mean": float(np.mean(vs_metrics)) if vs_metrics else None,
            "m_mean": float(np.mean(m_metrics)) if m_metrics else None,
            "n_epochs": len(self.measurements),
        }

    def find_stable_neighbors(
        self,
        radius_deg: float = 0.5,
        min_flux_ratio: float = 0.1,
        max_flux_ratio: float = 10.0,
        max_eta: float = 1.5,
        min_epochs: int = 10,
    ) -> list[str]:
        """Find stable neighbor sources suitable for relative photometry.

        Parameters
        ----------
        radius_deg : float
            Search radius in degrees (Default value = 0.5)
        min_flux_ratio : float
            Minimum neighbor flux ratio (relative to target) (Default value = 0.1)
        max_flux_ratio : float
            Maximum neighbor flux ratio (relative to target) (Default value = 10.0)
        max_eta : float
            Maximum eta metric (variability) for stable neighbors (Default value = 1.5)
        min_epochs : int
            Minimum number of observation epochs required (Default value = 10)

        """
        if not self.products_db or not self.products_db.exists():
            raise SourceError(f"Products database not found: {self.products_db}")

        # Get target median flux to filter neighbors by brightness
        if self.measurements.empty:
            return []

        flux_col = "peak_jyb"
        has_normalized = "normalized_flux_jy" in self.measurements.columns
        if has_normalized:
            flux_col = "normalized_flux_jy"
        target_flux = self.measurements[flux_col].median()
        if np.isnan(target_flux):
            return []

        conn = sqlite3.connect(str(self.products_db), timeout=30.0)
        conn.row_factory = sqlite3.Row

        try:
            # We need variability_stats table
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "variability_stats" not in tables:
                logger.warning("variability_stats table not found, cannot select stable neighbors")
                return []

            # Spatial query + variability filter + flux filter
            # Note: doing Haversine in SQLite is hard, so we do box filter + post-filter

            # Handle Pole Singularity
            cos_dec = np.cos(np.deg2rad(self.dec_deg))
            if abs(self.dec_deg) > 89.0 or abs(cos_dec) < 1e-6:
                # Near pole: search all RA
                min_ra, max_ra = 0.0, 360.0
                ra_clause = "(1=1)"  # Always true for RA
                ra_params = ()
            else:
                ra_range = radius_deg / cos_dec
                min_ra = self.ra_deg - ra_range
                max_ra = self.ra_deg + ra_range

                # Check for RA wrapping
                if min_ra < 0 or max_ra > 360:
                    # Wrapped case: ranges overlap 0/360 boundary
                    # We search (ra >= min_norm) OR (ra <= max_norm)
                    min_ra_norm = min_ra % 360
                    max_ra_norm = max_ra % 360
                    ra_clause = "(ra_deg >= ? OR ra_deg <= ?)"
                    ra_params = (min_ra_norm, max_ra_norm)
                else:
                    # Standard case
                    ra_clause = "(ra_deg BETWEEN ? AND ?)"
                    ra_params = (min_ra, max_ra)

            min_dec = max(-90.0, self.dec_deg - radius_deg)
            max_dec = min(90.0, self.dec_deg + radius_deg)

            query = f"""
                SELECT source_id, ra_deg, dec_deg, mean_flux_mjy, eta_metric
                FROM variability_stats
                WHERE {ra_clause}
                  AND dec_deg BETWEEN ? AND ?
                  AND eta_metric < ?
                  AND n_obs >= ?
                  AND source_id != ?
            """  # nosec B608 - ra_clause is built from constants, not user input

            params = ra_params + (min_dec, max_dec, max_eta, min_epochs, self.source_id)

            cursor = conn.execute(query, params)
            candidates = cursor.fetchall()

            if not candidates:
                return []

            target_flux_mjy = target_flux * 1000.0

            # Convert candidates to arrays for vectorized operations
            df = pd.DataFrame(candidates, columns=candidates[0].keys())
            ra_degs = df["ra_deg"].to_numpy()
            dec_degs = df["dec_deg"].to_numpy()

            # 1. Vectorized distance check
            cos_dec = np.cos(np.deg2rad(self.dec_deg))
            dists_sq = (
                ((ra_degs - self.ra_deg) * cos_dec) ** 2
                + (dec_degs - self.dec_deg) ** 2
            )
            dist_mask = dists_sq <= radius_deg**2

            # 2. Vectorized flux ratio check
            flux_ratios = df["mean_flux_mjy"].to_numpy() / target_flux_mjy
            flux_mask = (
                (flux_ratios >= min_flux_ratio)
                & (flux_ratios <= max_flux_ratio)
            )

            # Filter IDs by combined mask
            return df["source_id"].to_numpy()[
                dist_mask & flux_mask
            ].tolist()

        finally:
            conn.close()

    def calculate_relative_lightcurve(
        self,
        radius_deg: float = 0.5,
        min_flux_ratio: float = 0.1,
        max_flux_ratio: float = 10.0,
        max_eta: float = 1.5,
        min_neighbors: int = 1,
        max_neighbors: int = 20,
    ) -> dict[str, Any]:
        """Calculate relative lightcurve using stable neighbors.

            This automatically finds neighbors, loads their data, aligns timestamps,
            and calculates the relative flux metrics.

        Parameters
        ----------
        radius_deg : float
            Search radius for neighbors (Default value = 0.5)
        min_flux_ratio : float
            Minimum flux ratio for neighbors (Default value = 0.1)
        max_flux_ratio : float
            Maximum flux ratio for neighbors (Default value = 10.0)
        max_eta : float
            Stability threshold for neighbors (Default value = 1.5)
        min_neighbors : int
            Minimum number of neighbors required (Default value = 1)
        max_neighbors : int
            Maximum number of neighbors to use (closest/brightest) (Default value = 20)

        """
        from dsa110_contimg.core.photometry.variability import calculate_relative_flux

        # 1. Find Neighbors
        neighbor_ids = self.find_stable_neighbors(
            radius_deg=radius_deg,
            min_flux_ratio=min_flux_ratio,
            max_flux_ratio=max_flux_ratio,
            max_eta=max_eta,
        )

        if len(neighbor_ids) < min_neighbors:
            logger.warning(f"Found only {len(neighbor_ids)} neighbors, need {min_neighbors}")
            return {}

        # Limit to max_neighbors (could optimize to pick best ones, currently just first N)
        neighbor_ids = neighbor_ids[:max_neighbors]

        # 2. Load Neighbor Data
        # We need to align neighbor measurements to target epochs (by image_path or measured_at)
        # Assuming 'image_path' is unique per epoch

        if self.measurements.empty:
            return {}

        # Ensure required columns exist
        required_cols = ["image_path", "mjd", "normalized_flux_jy"]
        if not all(col in self.measurements.columns for col in required_cols):
            logger.warning(f"Missing required columns for relative photometry: {required_cols}")
            return {}

        target_epochs = self.measurements[["image_path", "mjd", "normalized_flux_jy"]].copy()
        target_epochs = target_epochs.set_index("image_path")

        neighbor_flux_matrix = []
        neighbor_error_matrix = []
        valid_neighbor_ids = []
        load_errors = []

        for nid in neighbor_ids:
            try:
                n_src = Source(nid, products_db=self.products_db)
                n_meas = n_src.measurements
                if n_meas.empty:
                    continue

                # Align with target
                # Join on image_path
                n_data = n_meas[
                    ["image_path", "normalized_flux_jy", "normalized_flux_err_jy"]
                ].set_index("image_path")

                # Reindex to target epochs (puts NaNs where missing)
                aligned = n_data.reindex(target_epochs.index)

                neighbor_flux_matrix.append(aligned["normalized_flux_jy"].values)
                neighbor_error_matrix.append(aligned["normalized_flux_err_jy"].values)
                valid_neighbor_ids.append(nid)

            except Exception as e:
                logger.warning(f"Error loading neighbor {nid}: {e}")
                load_errors.append(str(e))
                continue

        if not valid_neighbor_ids:
            if neighbor_ids and len(load_errors) == len(neighbor_ids):
                logger.error(
                    f"All {len(neighbor_ids)} neighbors failed to load. First error: {load_errors[0]}"
                )
            return {}

        # Transpose to (n_epochs, n_neighbors)
        neighbor_errors = np.array(neighbor_error_matrix).T

        # 3. Calculate Relative Flux
        rel_flux, mean_val, std_val = calculate_relative_flux(
            target_fluxes=target_epochs["normalized_flux_jy"].values,
            neighbor_fluxes=neighbor_fluxes,
            neighbor_errors=neighbor_errors,
            use_robust_stats=True,
        )

        return {
            "relative_flux": rel_flux,
            "relative_flux_mean": mean_val,
            "relative_flux_std": std_val,
            "n_neighbors": len(valid_neighbor_ids),
            "neighbor_ids": valid_neighbor_ids,
            "mjds": target_epochs["mjd"].values,
        }

    def plot_lightcurve(
        self,
        use_normalized: bool = True,
        figsize: tuple[int, int] = (10, 6),
        min_points: int = 2,
        mjd: bool = False,
        grid: bool = True,
        yaxis_start: str = "auto",
        highlight_baseline: bool = True,
        highlight_ese_period: bool = True,
        save: bool = False,
        outfile: str | None = None,
        plot_dpi: int = 150,
    ):
        """Plot light curve with ESE-specific features.

            Adopted from VAST Tools with DSA-110 specific enhancements:
            - Baseline period highlighting (first 10 epochs)
            - ESE candidate period highlighting (14-180 days)
            - Normalized flux plotting

        Parameters
        ----------
        use_normalized : bool
            Use normalized flux (default) or raw flux (Default value = True)
        figsize : Tuple[int, int]
            Figure size tuple (Default value = (10, 6))
        min_points : int
            Minimum number of points required (Default value = 2)
        mjd : bool
            Use MJD for x-axis instead of datetime (Default value = False)
        grid : bool
            Show grid (Default value = True)
        yaxis_start : str
            'auto' or '0' for y-axis start (Default value = "auto")
        highlight_baseline : bool
            Highlight first 10 epochs as baseline (Default value = True)
        highlight_ese_period : bool
            Highlight 14-180 day ESE candidate period (Default value = True)
        save : bool
            Save figure instead of returning (Default value = False)
        outfile : Optional[str]
            Output filename (auto-generated if None) (Default value = None)
        plot_dpi : int
            DPI for saved figure (Default value = 150)

        """
        import matplotlib.pyplot as plt

        if len(self.measurements) < min_points:
            raise SourceError(
                f"Need at least {min_points} measurements, have {len(self.measurements)}"
            )

        fig, ax = plt.subplots(figsize=figsize, dpi=plot_dpi)

        # Select flux column
        if use_normalized and "normalized_flux_jy" in self.measurements.columns:
            flux_col = "normalized_flux_jy"
            err_col = "normalized_flux_err_jy"
            flux_label = "Normalized Flux"
        else:
            flux_col = "peak_jyb"
            err_col = "peak_err_jyb"
            flux_label = "Peak Flux"

        if flux_col not in self.measurements.columns:
            raise SourceError(f"Flux column {flux_col} not found in measurements")

        # Time axis
        if mjd:
            if "mjd" not in self.measurements.columns:
                raise SourceError("MJD column not found in measurements")
            time = self.measurements["mjd"]
            xlabel = "MJD"
        else:
            if "measured_at" in self.measurements.columns:
                time = pd.to_datetime(self.measurements["measured_at"])
            elif "mjd" in self.measurements.columns:
                time = pd.to_datetime(
                    [Time(mjd, format="mjd").datetime for mjd in self.measurements["mjd"]]
                )
            else:
                raise SourceError("No time column found in measurements")
            xlabel = "Date"

        # Get flux and errors
        flux = self.measurements[flux_col]
        flux_err = self.measurements[err_col] if err_col in self.measurements.columns else None

        # Filter out NaN/inf
        valid_mask = np.isfinite(flux)
        if flux_err is not None:
            valid_mask &= np.isfinite(flux_err) & (flux_err > 0)

        if valid_mask.sum() < min_points:
            raise SourceError(f"Only {valid_mask.sum()} valid measurements")

        time_valid = time[valid_mask]
        flux_valid = flux[valid_mask]
        flux_err_valid = flux_err[valid_mask] if flux_err is not None else None

        # Plot flux with error bars
        if flux_err_valid is not None:
            ax.errorbar(
                time_valid,
                flux_valid,
                yerr=flux_err_valid,
                fmt="o",
                capsize=3,
                capthick=1.5,
                label=flux_label,
                markersize=6,
                alpha=0.7,
            )
        else:
            ax.plot(time_valid, flux_valid, "o", label=flux_label, markersize=6, alpha=0.7)

        # Highlight baseline period (first 10 epochs)
        if highlight_baseline and len(time_valid) >= 10:
            baseline_time = time_valid.iloc[:10]
            baseline_flux = flux_valid.iloc[:10]
            ax.axvspan(
                baseline_time.min(),
                baseline_time.max(),
                alpha=0.2,
                color="green",
                label="Baseline Period (first 10 epochs)",
            )
            # Plot baseline median
            baseline_median = baseline_flux.median()
            ax.axhline(
                baseline_median,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Baseline Median: {baseline_median:.4f} Jy",
            )

        # Highlight ESE candidate period (14-180 days)
        if highlight_ese_period and len(time_valid) > 1:
            if mjd:
                time_range_days = float(time_valid.max() - time_valid.min())
            else:
                time_range_days = (time_valid.max() - time_valid.min()).total_seconds() / 86400

            if 14 <= time_range_days <= 180:
                ax.axvspan(
                    time_valid.min(),
                    time_valid.max(),
                    alpha=0.1,
                    color="red",
                    label="ESE Candidate Period (14-180 days)",
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Flux (Jy)" if not use_normalized else "Normalized Flux")
        ax.set_title(f"Light Curve: {self.name}")
        ax.grid(grid, alpha=0.3)

        if yaxis_start == "0":
            ax.set_ylim(bottom=0)

        ax.legend(loc="best")

        plt.tight_layout()

        if save:
            if outfile is None:
                safe_name = self.source_id.replace(" ", "_").replace("/", "_")
                outfile = f"{safe_name}_lightcurve.png"
            fig.savefig(outfile, dpi=plot_dpi, bbox_inches="tight")
            logger.info(f"Saved light curve to {outfile}")
            plt.close(fig)
            return None

        return fig

    def crossmatch_external(
        self,
        radius_arcsec: float = 5.0,
        catalogs: list[str] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, dict[str, Any]]:
        """Cross-match this source with external catalogs.

            Queries external astronomical catalogs to identify the source and retrieve
            additional information. Uses the dsa110_contimg.core.catalog.multiwavelength module
            (ported from vast-mw).

        Parameters
        ----------
        radius_arcsec : float
            Search radius in arcseconds (default: 5.0) (Default value = 5.0)
        catalogs : Optional[List[str]]
            List of catalogs to query. Options: 'Gaia', 'Simbad', 'ATNF', 'NVSS', etc.
            If None, queries all configured services. (Default value = None)
        timeout : float
            Query timeout in seconds (not all services support this) (Default value = 30.0)

        Returns
        -------
            >>> source = Source(source_id="NVSS J123456+012345")
            >>> results = source.crossmatch_external(radius_arcsec=10.0)
            >>> if 'Simbad' in results:
            ...     print(f"Matches: {results['Simbad']}")
        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        if self.ra_deg is None or self.dec_deg is None:
            logger.warning(f"Cannot cross-match source {self.source_id}: RA/Dec not available")
            return {}

        # Use J2000 epoch for static catalogs, or apply proper motion if time available
        if not self.measurements.empty and "mjd" in self.measurements.columns:
            t = Time(self.measurements["mjd"].iloc[0], format="mjd")
        else:
            t = None

        coord = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg, obstime=t)

        if catalogs is None:
            return check_all_services(coord, t=t, radius=radius_arcsec * u.arcsec)

        results = {}
        services = {
            "Gaia": check_gaia,
            "Simbad": check_simbad,
            "ATNF": check_atnf,
            "NVSS": check_nvss,
            "FIRST": check_first,
            "Pulsar Scraper": check_pulsarscraper,
        }

        for cat in catalogs:
            if cat in services:
                func = services[cat]
                try:
                    if func in [check_gaia, check_simbad, check_atnf]:
                        res = func(coord, t=t, radius=radius_arcsec * u.arcsec)
                    else:
                        res = func(coord, radius=radius_arcsec * u.arcsec)
                    if res:
                        results[cat] = res
                except Exception as e:
                    logger.warning(f"Crossmatch for {cat} failed: {e}")
            else:
                logger.warning(f"Catalog {cat} not supported")

        return results

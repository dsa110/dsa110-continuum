"""ImagingParams: Parameter object for imaging functions.

This module provides a dataclass for bundling imaging parameters,
reducing function signature complexity from 30+ kwargs to a single
configuration object.

Examples
--------
>>> params = ImagingParams(
...     imagename="output",
...     imsize=2048,
...     cell_arcsec=1.5,
...     niter=5000,
... )
>>> # Use with image_ms
>>> image_ms_with_params(ms_path, params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ImagingParams:
    """Configuration parameters for imaging operations.

    This dataclass bundles the numerous imaging parameters into a single
    object, making function signatures cleaner and configuration more
    manageable.

    """

    # Required parameter
    imagename: str

    # Field/selection parameters
    field: str = ""
    spw: str = ""

    # Image geometry
    imsize: int = 2400
    cell_arcsec: float | Literal["auto"] = 6.0

    # Weighting
    weighting: str = "briggs"
    robust: float = 0.5

    # Spectral configuration
    specmode: str = "mfs"
    deconvolver: str = "hogbom"
    nterms: int = 1

    # Cleaning parameters
    niter: int = 1000
    threshold: str = "0.005Jy"
    auto_mask: float = 5.0  # Auto-mask threshold in sigma (5σ for transient)
    auto_threshold: float = 1.0  # Stop cleaning at this sigma level
    mgain: float = 0.8  # Major cycle gain

    # Primary beam
    pbcor: bool = True
    pblimit: float = 0.2
    psfcutoff: float | None = None

    # Coordinate/gridding
    phasecenter: str | None = None
    gridder: str = "idg"
    wprojplanes: int = -1
    uvrange: str = ">1klambda"

    # Quality and output
    quality_tier: str = "standard"
    skip_fits: bool = False

    # Advanced options
    vptable: str | None = None
    wbawp: bool | None = None
    cfcache: str | None = None

    # Catalog seeding
    unicat_min_mjy: float | None = None
    nvss_min_mjy: float | None = None
    calib_ra_deg: float | None = None
    calib_dec_deg: float | None = None
    calib_flux_jy: float | None = None

    # Backend configuration
    backend: str = "wsclean"
    wsclean_path: str | None = None
    export_model_image: bool = False

    # Masking
    use_unicat_mask: bool = True
    mask_path: str | None = None
    mask_radius_arcsec: float = 60.0
    target_mask: str | None = None
    galvin_clip_mask: str | None = None
    galvin_box_size: int = 100
    galvin_adaptive_depth: int = 3
    erode_beam_shape: bool = False

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Validate weighting
        valid_weightings = {"briggs", "natural", "uniform"}
        if self.weighting not in valid_weightings:
            raise ValueError(f"weighting must be one of {valid_weightings}, got '{self.weighting}'")

        # Validate robust range
        if not -2.0 <= self.robust <= 2.0:
            raise ValueError(f"robust must be between -2 and 2, got {self.robust}")

        # Validate quality tier
        valid_tiers = {"development", "standard", "high_precision"}
        if self.quality_tier not in valid_tiers:
            raise ValueError(
                f"quality_tier must be one of {valid_tiers}, got '{self.quality_tier}'"
            )

        # Validate backend
        valid_backends = {"wsclean", "tclean"}
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got '{self.backend}'")

        # Validate cell_arcsec
        if self.cell_arcsec != "auto":
            if not isinstance(self.cell_arcsec, (int, float)):
                raise ValueError(
                    f"cell_arcsec must be a number or 'auto', got '{self.cell_arcsec}'"
                )
            if self.cell_arcsec <= 0:
                raise ValueError(f"cell_arcsec must be positive, got {self.cell_arcsec}")

        # Validate deconvolver
        valid_deconvolvers = {"hogbom", "mtmfs", "clark", "multiscale"}
        if self.deconvolver not in valid_deconvolvers:
            raise ValueError(
                f"deconvolver must be one of {valid_deconvolvers}, got '{self.deconvolver}'"
            )

        # Validate positive values
        if self.imsize <= 0:
            raise ValueError(f"imsize must be positive, got {self.imsize}")
        if self.niter < 0:
            raise ValueError(f"niter must be non-negative, got {self.niter}")
        if self.nterms < 1:
            raise ValueError(f"nterms must be >= 1, got {self.nterms}")
        if self.mask_radius_arcsec <= 0:
            raise ValueError(f"mask_radius_arcsec must be positive, got {self.mask_radius_arcsec}")

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to dictionary for function calls.

        Returns
        -------
            Dictionary of imaging parameters suitable for **kwargs expansion.

        """
        return {
            "imagename": self.imagename,
            "field": self.field,
            "spw": self.spw,
            "imsize": self.imsize,
            "cell_arcsec": self.cell_arcsec,
            "weighting": self.weighting,
            "robust": self.robust,
            "specmode": self.specmode,
            "deconvolver": self.deconvolver,
            "nterms": self.nterms,
            "niter": self.niter,
            "threshold": self.threshold,
            "auto_mask": self.auto_mask,
            "auto_threshold": self.auto_threshold,
            "mgain": self.mgain,
            "pbcor": self.pbcor,
            "phasecenter": self.phasecenter,
            "gridder": self.gridder,
            "wprojplanes": self.wprojplanes,
            "uvrange": self.uvrange,
            "pblimit": self.pblimit,
            "psfcutoff": self.psfcutoff,
            "quality_tier": self.quality_tier,
            "skip_fits": self.skip_fits,
            "vptable": self.vptable,
            "wbawp": self.wbawp,
            "cfcache": self.cfcache,
            "unicat_min_mjy": self.unicat_min_mjy,
            "nvss_min_mjy": self.nvss_min_mjy,
            "calib_ra_deg": self.calib_ra_deg,
            "calib_dec_deg": self.calib_dec_deg,
            "calib_flux_jy": self.calib_flux_jy,
            "backend": self.backend,
            "wsclean_path": self.wsclean_path,
            "export_model_image": self.export_model_image,
            "use_unicat_mask": self.use_unicat_mask,
            "mask_path": self.mask_path,
            "mask_radius_arcsec": self.mask_radius_arcsec,
            "target_mask": self.target_mask,
            "galvin_clip_mask": self.galvin_clip_mask,
            "galvin_box_size": self.galvin_box_size,
            "galvin_adaptive_depth": self.galvin_adaptive_depth,
            "erode_beam_shape": self.erode_beam_shape,
        }

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ImagingParams:
        """Create ImagingParams from dictionary.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters.

        Returns
        -------
            ImagingParams
            ImagingParams instance.
        """
        return cls(**params)

    def with_overrides(self, **kwargs: Any) -> ImagingParams:
        """Create a new ImagingParams with specified overrides.

        Parameters
        ----------
        **kwargs : Any
            Parameters to override

        Returns
        -------
        ImagingParams
            New ImagingParams instance with overrides applied

        Examples
        --------
        >>> base = ImagingParams(imagename="test", niter=1000)
        >>> modified = base.with_overrides(niter=5000, robust=0.5)
        """
        current = self.to_dict()
        current.update(kwargs)
        return ImagingParams.from_dict(current)

    @classmethod
    def for_development(cls, imagename: str, **kwargs: Any) -> ImagingParams:
        """Create development-tier params for quick testing.

        Development tier uses coarser resolution and fewer iterations
        for faster (non-science quality) imaging.

        Parameters
        ----------
        imagename :
            Output image name
        **kwargs :
            Additional overrides

        Returns
        -------
            ImagingParams configured for development

        """
        defaults = {
            "imagename": imagename,
            "quality_tier": "development",
            "niter": 300,
            "imsize": 512,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_standard(cls, imagename: str, **kwargs: Any) -> ImagingParams:
        """Create standard-tier params for production imaging.

        Parameters
        ----------
        imagename :
            Output image name
        **kwargs :
            Additional overrides

        Returns
        -------
            ImagingParams configured for standard quality

        """
        defaults = {
            "imagename": imagename,
            "quality_tier": "standard",
            "niter": 1000,
            "imsize": 2400,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_high_precision(cls, imagename: str, **kwargs: Any) -> ImagingParams:
        """Create high-precision params for science-quality imaging.

        High precision uses more iterations and finer thresholds
        for the best image quality.

        Parameters
        ----------
        imagename :
            Output image name
        **kwargs :
            Additional overrides

        Returns
        -------
            ImagingParams configured for high precision

        """
        defaults = {
            "imagename": imagename,
            "quality_tier": "high_precision",
            "niter": 5000,
            "imsize": 2400,
            "threshold": "0.05mJy",
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_survey(cls, imagename: str, **kwargs: Any) -> ImagingParams:
        """Create params for DSA-110 daily survey imaging (primary science mode).

        The DSA-110 continuum pipeline operates in survey mode: daily mosaics
        are created and photometry is performed on ~10⁴ compact sources to
        detect Extreme Scattering Events (ESEs) via flux variability.

        Survey mode settings:
        - Multiscale deconvolver: Recovers both point and extended sources
        - nterms=2: Spectral index fitting for 18% fractional bandwidth
        - Deep cleaning: 10000 iterations for mosaic quality
        - Briggs robust=0: Balanced resolution/sensitivity

        DSA-110 context:
        - Transit instrument: ~5 min tiles as sources drift through beam
        - No tracking: Can only slew in elevation
        - Mosaic depth: ~3 overlapping tiles in overlap regions
        - Daily cadence: Photometry on mosaics to detect 2× flux drops

        Adjustable parameters (via kwargs):
        - niter: Cleaning depth (default 10000)
        - robust: Resolution/sensitivity tradeoff (-2 to 2, default 0)
        - deconvolver: "hogbom" for speed, "multiscale" for quality
        - nterms: 1 for speed, 2 for spectral index

        Parameters
        ----------
        imagename :
            Output image name
        **kwargs :
            Additional overrides

        Returns
        -------
            ImagingParams configured for survey imaging

        """
        defaults = {
            "imagename": imagename,
            "quality_tier": "high_precision",
            "deconvolver": "multiscale",  # Extended source recovery
            "nterms": 2,  # Spectral index for 18% fractional bandwidth
            "niter": 10000,  # Deep cleaning for mosaic quality
            "auto_mask": 4.0,  # Aggressive masking for clean mosaics
            "auto_threshold": 0.5,  # Deep cleaning threshold
            "imsize": 2400,
            "robust": 0.0,  # Briggs 0 for balanced resolution
        }
        defaults.update(kwargs)
        return cls(**defaults)


def image_ms_with_params(ms_path: str, params: ImagingParams) -> None:
    """Image a Measurement Set using ImagingParams.

        This is a convenience wrapper that accepts an ImagingParams object
        instead of individual keyword arguments.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    params : ImagingParams
        Imaging parameters object.

    Returns
    -------
        None

    Examples
    --------
        >>> params = ImagingParams.for_standard("output_image")
        >>> image_ms_with_params("/data/obs.ms", params)
    """
    from dsa110_contimg.core.imaging.cli_imaging import image_ms

    image_ms(ms_path, **params.to_dict())

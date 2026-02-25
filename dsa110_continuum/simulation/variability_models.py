"""Variability models for time-domain synthetic data generation.

    This module provides models for simulating time-varying radio sources including:
    - Constant flux (baseline)
    - Flare events (fast rise, exponential decay)
    - ESE (Extreme Scattering Events) - deep flux dips
    - Periodic variations (pulsars, binaries)

    These models are used by the time-domain simulation to inject realistic
    variability into synthetic visibilities, enabling validation of the pipeline's
    lightcurve and transient detection capabilities.

    Example
-------
    >>> from dsa110_contimg.core.simulation.variability_models import FlareModel
    >>> flare = FlareModel(
    ...     peak_time_mjd=60000.5,
    ...     rise_time_hours=0.5,
    ...     decay_time_hours=2.0,
    ...     peak_flux_jy=5.0,
    ...     baseline_flux_jy=1.0,
    ... )
    >>> flux_at_peak = flare.evaluate(60000.5)  # Returns 5.0 Jy
    >>> flux_after_decay = flare.evaluate(60000.5 + 4/24)  # After 4 hours
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Hours to days conversion
HOURS_TO_DAYS = 1.0 / 24.0


@dataclass
class VariabilityModel(ABC):
    """Base class for all variability models.

    All subclasses must implement evaluate() to return flux at a given time.

    """

    model_type: str = "base"  # Must have default since subclasses have defaults
    baseline_flux_jy: float = 0.0  # Must have default since subclasses have defaults

    @abstractmethod
    def evaluate(self, mjd: float) -> float:
        """Compute flux density at the specified MJD.

        Parameters
        ----------
        mjd :
            Modified Julian Date

        Returns
        -------
            Flux density in Jy

        """
        pass

    def to_dict(self) -> dict:
        """Serialize model to dictionary for metadata storage."""
        result = {"model_type": self.model_type, "baseline_flux_jy": self.baseline_flux_jy}
        # Add all other fields from dataclass
        for field, value in self.__dict__.items():
            if field not in result:
                result[field] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> VariabilityModel:
        """Deserialize model from dictionary."""
        model_type = data.get("model_type")
        if model_type == "constant":
            return ConstantFlux(**data)
        elif model_type == "flare":
            return FlareModel(**data)
        elif model_type == "ese":
            return ESEScattering(**data)
        elif model_type == "periodic":
            return PeriodicVariation(**data)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


@dataclass
class ConstantFlux(VariabilityModel):
    """Constant flux - no variability (baseline for testing).

        Example
    -------
        >>> constant = ConstantFlux(baseline_flux_jy=2.5)
        >>> constant.evaluate(60000.0)  # Always returns 2.5 Jy
        2.5
    """

    model_type: str = "constant"

    def evaluate(self, mjd: float) -> float:
        """
        Parameters
        ----------
        mjd : float
            The modified Julian date to evaluate.
        """
        return self.baseline_flux_jy


@dataclass
class FlareModel(VariabilityModel):
    """Radio flare with fast rise and exponential decay.

    Models typical radio flare behavior:
    - Fast rise to peak (linear)
    - Exponential decay back to baseline

    Flux profile:
    - Before t_peak - rise_time: baseline
    - Rising phase: linear interpolation
    - At t_peak: peak_flux_jy
    - Decay phase: exponential decay
    - After decay complete: baseline

    Parameters
    ----------
    peak_time_mjd : float
        Time of maximum flux (MJD)
    rise_time_hours : float
        Time from onset to peak (hours)
    decay_time_hours : float
        e-folding time for exponential decay (hours)
    peak_flux_jy : float
        Maximum flux density (Jy)
    baseline_flux_jy : float
        Quiescent flux density (Jy)

    Examples
    --------
    >>> flare = FlareModel(
    ...     peak_time_mjd=60000.5,
    ...     rise_time_hours=1.0,
    ...     decay_time_hours=3.0,
    ...     peak_flux_jy=10.0,
    ...     baseline_flux_jy=2.0,
    ... )
    >>> flare.evaluate(60000.5)  # At peak
    10.0
    >>> flare.evaluate(60000.5 + 3/24)  # After 3 hours (1 decay constant)
    4.943...  # baseline + (peak - baseline) * exp(-1)
    """

    model_type: str = "flare"
    peak_time_mjd: float = 0.0
    rise_time_hours: float = 1.0
    decay_time_hours: float = 2.0
    peak_flux_jy: float = 5.0

    def evaluate(self, mjd: float) -> float:
        """Compute flux at given MJD using rise/decay model."""
        rise_time_days = self.rise_time_hours * HOURS_TO_DAYS
        decay_time_days = self.decay_time_hours * HOURS_TO_DAYS

        t_start = self.peak_time_mjd - rise_time_days
        dt = mjd - t_start

        # Before flare starts
        if dt < 0:
            return self.baseline_flux_jy

        # Rising phase (linear)
        if dt < rise_time_days:
            fraction = dt / rise_time_days
            return self.baseline_flux_jy + (self.peak_flux_jy - self.baseline_flux_jy) * fraction

        # Decay phase (exponential)
        dt_decay = mjd - self.peak_time_mjd
        amplitude = self.peak_flux_jy - self.baseline_flux_jy
        decay_factor = math.exp(-dt_decay / decay_time_days)
        return self.baseline_flux_jy + amplitude * decay_factor


@dataclass
class ESEScattering(VariabilityModel):
    """Extreme Scattering Event - deep flux dip with recovery.

        Models ESE behavior observed in compact radio sources:
        - Gradual flux decrease to minimum
        - Deep flux dip (factor of 2-10x)
        - Gradual recovery to baseline

        The profile uses symmetric Gaussian dips for simplicity.

    Parameters
    ----------
    dip_time_mjd : float
        Time of minimum flux (MJD)
    dip_duration_days : float
        Full width at half maximum (FWHM) of the dip in days
    dip_depth_factor : float
        Flux reduction factor (e.g., 0.1 = 90% flux loss)
    baseline_flux_jy : float
        Quiescent flux density (Jy)

    Examples
    --------
        >>> ese = ESEScattering(
        ...     dip_time_mjd=60000.0,
        ...     dip_duration_days=7.0,
        ...     dip_depth_factor=0.2,  # 80% flux loss at minimum
        ...     baseline_flux_jy=3.0,
        ... )
        >>> ese.evaluate(60000.0)  # At dip minimum
        0.6  # 3.0 * 0.2
        >>> ese.evaluate(60000.0 + 10)  # Well after dip
        3.0  # Back to baseline
    """

    model_type: str = "ese"
    dip_time_mjd: float = 0.0
    dip_duration_days: float = 7.0
    dip_depth_factor: float = 0.1  # Minimum flux = baseline * factor

    def evaluate(self, mjd: float) -> float:
        """Compute flux using Gaussian dip profile."""
        # Use Gaussian profile with sigma = duration / 2.355 (FWHM relation)
        sigma_days = self.dip_duration_days / 2.355
        dt = mjd - self.dip_time_mjd

        # Gaussian dip: 1 - (1 - dip_factor) * exp(-dt^2 / (2*sigma^2))
        dip_amplitude = 1.0 - self.dip_depth_factor
        gaussian = math.exp(-0.5 * (dt / sigma_days) ** 2)
        flux_factor = 1.0 - dip_amplitude * gaussian

        return self.baseline_flux_jy * flux_factor


@dataclass
class PeriodicVariation(VariabilityModel):
    """Periodic flux variation (sinusoidal).

        Models periodic sources like pulsars (when detectable) or binary systems.
        Uses simple sinusoidal modulation.

    Parameters
    ----------
    period_days : float
        Period of variation in days
    amplitude_jy : float
        Peak-to-peak amplitude in Jy
    phase_offset : float
        Phase offset (0 to 1)
    baseline_flux_jy : float
        Mean flux density in Jy

    Examples
    --------
        >>> periodic = PeriodicVariation(
        ...     period_days=1.0,
        ...     amplitude_jy=0.5,
        ...     phase_offset=0.0,
        ...     baseline_flux_jy=2.0,
        ... )
        >>> periodic.evaluate(60000.0)  # At phase 0
        2.0  # Mean flux
        >>> periodic.evaluate(60000.25)  # At phase 0.25 (quarter period)
        2.25  # Peak = baseline + amplitude/2
    """

    model_type: str = "periodic"
    period_days: float = 1.0
    amplitude_jy: float = 0.5
    phase_offset: float = 0.0

    def evaluate(self, mjd: float) -> float:
        """Compute flux using sinusoidal modulation."""
        phase = ((mjd / self.period_days) + self.phase_offset) % 1.0
        modulation = math.sin(2 * math.pi * phase)
        return self.baseline_flux_jy + 0.5 * self.amplitude_jy * modulation


def compute_flux_at_time(
    baseline_flux_jy: float,
    model: VariabilityModel | None,
    mjd: float,
) -> float:
    """Compute flux density at specified time given a variability model.

    Parameters
    ----------
    baseline_flux_jy : float
        Flux when model is None or constant
    model : Optional[VariabilityModel]
        Variability model (None = constant)
    mjd : float
        Modified Julian Date

    Returns
    -------
    float
        Flux density in Jy

    Examples
    --------
    >>> flare = FlareModel(peak_time_mjd=60000.0, peak_flux_jy=5.0, baseline_flux_jy=1.0)
    >>> compute_flux_at_time(1.0, flare, 60000.0)
    5.0
    >>> compute_flux_at_time(1.0, None, 60000.0)
    1.0
    """
    if model is None:
        return baseline_flux_jy
    return model.evaluate(mjd)


def create_variability_model(
    model_type: Literal["constant", "flare", "ese", "periodic"],
    baseline_flux_jy: float,
    **params: Any,
) -> VariabilityModel:
    """Factory function to create variability models.

    Parameters
    ----------
    model_type : Literal["constant", "flare", "ese", "periodic"]
        Type of model to create.
    baseline_flux_jy : float
        Quiescent flux density (Jy).
        **params : dict
        Model-specific parameters.

    Returns
    -------
    model : VariabilityModel
        Initialized variability model.

    Examples
    --------
        >>> model = create_variability_model(
        ...     "flare",
        ...     baseline_flux_jy=2.0,
        ...     peak_time_mjd=60000.0,
        ...     peak_flux_jy=10.0,
        ... )
        >>> isinstance(model, FlareModel)
        True
    """
    if model_type == "constant":
        return ConstantFlux(baseline_flux_jy=baseline_flux_jy)
    elif model_type == "flare":
        return FlareModel(baseline_flux_jy=baseline_flux_jy, **params)
    elif model_type == "ese":
        return ESEScattering(baseline_flux_jy=baseline_flux_jy, **params)
    elif model_type == "periodic":
        return PeriodicVariation(baseline_flux_jy=baseline_flux_jy, **params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

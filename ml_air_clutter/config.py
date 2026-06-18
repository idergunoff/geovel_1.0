from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class NormalizationConfig:
    """Serializable configuration for profile normalization."""

    mode: str = "standard"
    eps: float = 1e-8
    clip_lower_percentile: float = 1.0
    clip_upper_percentile: float = 99.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SyntheticClutterConfig:
    """Serializable configuration for analytical synthetic air-clutter generation."""

    seed: Optional[int] = 42
    target_snr_db: Optional[float] = 6.0
    hyperbolas: bool = True
    sloped_events: bool = True
    ringing: bool = True
    vertical_spikes: bool = True
    noise_zones: bool = True
    num_hyperbolas: int = 3
    num_sloped_events: int = 2
    num_ringing_events: int = 2
    num_vertical_spikes: int = 4
    num_noise_zones: int = 2
    base_amplitude: float = 1.0

    def to_dict(self):
        return asdict(self)

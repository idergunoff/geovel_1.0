from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NormalizationConfig:
    """Serializable configuration for profile normalization."""

    mode: str = "standard"
    eps: float = 1e-8
    clip_lower_percentile: float = 1.0
    clip_upper_percentile: float = 99.0

    def to_dict(self):
        return asdict(self)

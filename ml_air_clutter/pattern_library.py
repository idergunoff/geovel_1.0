import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .noise_patterns import DEFAULT_PATTERN_TAGS, pattern_statistics


@dataclass
class NoisePattern:
    pattern_id: str
    source_profile: str
    array: np.ndarray
    mask: np.ndarray
    bbox: list
    normalization: dict
    stats: dict
    tags: list = field(default_factory=lambda: ["unknown"])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    comment: str = ""

    @classmethod
    def create(cls, source_profile, array, mask, bbox, normalization=None, tags=None, comment="", pattern_id=None):
        pattern_tags = list(tags or ["unknown"])
        invalid_tags = sorted(set(pattern_tags) - set(DEFAULT_PATTERN_TAGS))
        if invalid_tags:
            raise ValueError(f"Unsupported pattern tags: {invalid_tags}")
        arr = np.asarray(array, dtype=float)
        m = (np.asarray(mask, dtype=float) > 0).astype(float)
        if arr.ndim != 2:
            raise ValueError("Pattern array must be 2D.")
        if m.shape != arr.shape:
            raise ValueError(f"Pattern mask shape {m.shape} must match array shape {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("Pattern array contains non-finite values.")
        return cls(
            pattern_id=pattern_id or str(uuid.uuid4()),
            source_profile=str(source_profile),
            array=arr,
            mask=m,
            bbox=[int(v) for v in bbox],
            normalization=normalization or {},
            stats=pattern_statistics(arr, m),
            tags=pattern_tags,
            comment=comment,
        )

    def to_metadata(self, array_file, mask_file):
        return {
            "pattern_id": self.pattern_id,
            "source_profile": self.source_profile,
            "array_file": array_file,
            "mask_file": mask_file,
            "bbox": self.bbox,
            "normalization": self.normalization,
            "stats": self.stats,
            "tags": self.tags,
            "created_at": self.created_at,
            "comment": self.comment,
        }


class PatternLibrary:
    """In-memory catalog of real air-clutter patterns with JSON/NPY persistence."""

    def __init__(self, patterns=None):
        self.patterns = list(patterns or [])

    def add_pattern(self, pattern):
        if self.get(pattern.pattern_id) is not None:
            raise ValueError(f"Pattern id already exists: {pattern.pattern_id}")
        self.patterns.append(pattern)
        return pattern

    def get(self, pattern_id):
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def summary(self):
        tags = {}
        for pattern in self.patterns:
            for tag in pattern.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {"num_patterns": len(self.patterns), "tags": tags}

    def save(self, directory):
        root = Path(directory)
        arrays_dir = root / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        index = {"version": 1, "saved_at": datetime.now(timezone.utc).isoformat(), "patterns": []}
        for pattern in self.patterns:
            array_file = f"arrays/{pattern.pattern_id}_array.npy"
            mask_file = f"arrays/{pattern.pattern_id}_mask.npy"
            np.save(root / array_file, pattern.array)
            np.save(root / mask_file, pattern.mask)
            index["patterns"].append(pattern.to_metadata(array_file, mask_file))
        with (root / "pattern_library_index.json").open("w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2)
        return root / "pattern_library_index.json"

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        index_path = root / "pattern_library_index.json"
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        patterns = []
        for meta in index.get("patterns", []):
            array = np.load(root / meta["array_file"])
            mask = np.load(root / meta["mask_file"])
            pattern = NoisePattern.create(
                source_profile=meta["source_profile"],
                array=array,
                mask=mask,
                bbox=meta["bbox"],
                normalization=meta.get("normalization", {}),
                tags=meta.get("tags", ["unknown"]),
                comment=meta.get("comment", ""),
                pattern_id=meta["pattern_id"],
            )
            pattern.created_at = meta.get("created_at", pattern.created_at)
            pattern.stats = meta.get("stats", pattern.stats)
            patterns.append(pattern)
        return cls(patterns)

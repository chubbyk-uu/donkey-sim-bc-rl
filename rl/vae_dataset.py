from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VaeMemmapDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str = "train", max_samples: int | None = None):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        manifest_path = Path(self.meta["manifest"])
        self.rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.indices = [idx for idx, row in enumerate(self.rows) if row.get("split") == split]
        if max_samples is not None and max_samples > 0:
            self.indices = self.indices[:max_samples]

        self._images = None

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def images(self) -> np.memmap:
        if self._images is None:
            self._images = np.memmap(
                self.meta["cache"],
                mode="r",
                dtype=np.uint8,
                shape=tuple(self.meta["shape"]),
            )
        return self._images

    def __getitem__(self, item: int) -> torch.Tensor:
        idx = self.indices[item]
        image = np.asarray(self.images[idx], dtype=np.uint8).copy()
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

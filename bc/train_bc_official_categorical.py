import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from train_bc import count_parameters, set_seed, split_indices, worker_init_fn


@dataclass
class TrainConfig:
    data_dir: list[str]
    val_dir: list[str] | None
    output_dir: str
    drop_head: int
    drop_tail: int
    cache_images: bool
    memmap_cache_dir: str | None
    min_throttle: float | None
    max_abs_angle: float | None
    steering_bins: int
    steering_min: float
    steering_max: float
    throttle_bins: int
    throttle_min: float
    throttle_max: float
    flip_prob: float
    sampler: str
    sampler_weight_max: float
    throttle_loss_weight: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    val_split: float
    seed: int
    init_from_regression: str | None


def make_bin_centers(num_bins: int, min_value: float, max_value: float) -> np.ndarray:
    edges = np.linspace(min_value, max_value, num_bins + 1, dtype=np.float32)
    return (edges[:-1] + edges[1:]) / 2.0


def closest_bin(value: float, centers: np.ndarray) -> int:
    clipped = float(np.clip(value, centers[0], centers[-1]))
    return int(np.argmin(np.abs(centers - clipped)))


class OfficialCategoricalDonkeyDataset(Dataset):
    def __init__(
        self,
        data_dirs: list[Path],
        steering_centers: np.ndarray,
        throttle_centers: np.ndarray,
        drop_head: int = 0,
        drop_tail: int = 0,
        cache_images: bool = False,
        memmap_cache_dir: Path | None = None,
        min_throttle: float | None = None,
        max_abs_angle: float | None = None,
        flip_prob: float = 0.0,
    ):
        if cache_images and memmap_cache_dir is not None:
            raise ValueError("--cache-images and --memmap-cache-dir are mutually exclusive")
        self.steering_centers = steering_centers
        self.throttle_centers = throttle_centers
        self.flip_prob = flip_prob
        self.records_data: list[dict] = []
        self.sample_indices: list[int] = []
        self.tub_summaries: list[dict] = []
        self.filtered_count = 0
        self.cache_images = cache_images
        self.memmap_cache_dir = memmap_cache_dir

        for road_id, data_dir in enumerate(data_dirs):
            records = sorted(data_dir.glob("record_*.json"), key=lambda p: int(p.stem.split("_")[1]))
            end = len(records) - drop_tail if drop_tail else len(records)
            records = records[drop_head:end]
            if not records:
                raise ValueError(f"no records selected in {data_dir}")

            tub_start = len(self.records_data)
            samples_before = len(self.sample_indices)
            angles = []
            throttles = []

            for record_path in records:
                record = json.loads(record_path.read_text())
                record["_data_dir"] = str(data_dir)
                record["_road_id"] = road_id
                record["_record_path"] = str(record_path)
                self.records_data.append(record)

            for local_index, record in enumerate(self.records_data[tub_start:]):
                angle = float(record["user/angle"])
                throttle = float(record["user/throttle"])
                if min_throttle is not None and throttle < min_throttle:
                    self.filtered_count += 1
                    continue
                if max_abs_angle is not None and abs(angle) > max_abs_angle:
                    self.filtered_count += 1
                    continue
                angles.append(angle)
                throttles.append(throttle)
                self.sample_indices.append(tub_start + local_index)

            angle_arr = np.asarray(angles, dtype=np.float32)
            throttle_arr = np.asarray(throttles, dtype=np.float32)
            if len(angle_arr) == 0:
                raise ValueError(f"all records filtered out in {data_dir}")
            self.tub_summaries.append(
                {
                    "road_id": road_id,
                    "data_dir": str(data_dir),
                    "records": len(records),
                    "samples": len(self.sample_indices) - samples_before,
                    "angle_min": float(angle_arr.min()),
                    "angle_max": float(angle_arr.max()),
                    "angle_mean": float(angle_arr.mean()),
                    "angle_std": float(angle_arr.std()),
                    "throttle_min": float(throttle_arr.min()),
                    "throttle_max": float(throttle_arr.max()),
                    "throttle_mean": float(throttle_arr.mean()),
                }
            )

        if not self.sample_indices:
            raise ValueError("no samples selected")
        self.image_shape = (3, 120, 160)
        self.image_cache = None
        self.image_memmap = None
        if memmap_cache_dir is not None:
            self.image_memmap = self._load_or_create_memmap_cache(memmap_cache_dir)
        elif cache_images:
            self.image_cache = self._load_image_cache()

    def _load_image_cache(self) -> list[np.ndarray]:
        return [self._load_image(record) for record in self.records_data]

    def _load_image(self, record: dict) -> np.ndarray:
        image_path = Path(record["_data_dir"]) / record["cam/image_array"]
        image = Image.open(image_path).convert("RGB")
        image_arr = np.array(image, dtype=np.uint8, copy=True)
        return np.transpose(image_arr, (2, 0, 1))

    def _cache_signature(self) -> str:
        payload = "\n".join(record["_record_path"] for record in self.records_data).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_or_create_memmap_cache(self, cache_dir: Path) -> np.memmap:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "meta.json"
        records_path = cache_dir / "records.json"
        images_path = cache_dir / "images.dat"
        expected_records = [record["_record_path"] for record in self.records_data]
        expected_meta = {
            "version": 1,
            "dtype": "uint8",
            "shape": [len(self.records_data), *self.image_shape],
            "signature": self._cache_signature(),
        }

        if meta_path.exists() and records_path.exists() and images_path.exists():
            meta = json.loads(meta_path.read_text())
            records = json.loads(records_path.read_text())
            if meta != expected_meta or records != expected_records:
                raise ValueError(
                    f"memmap cache at {cache_dir} does not match selected records; "
                    "delete it or use a different --memmap-cache-dir"
                )
        else:
            print(f"building image memmap cache: {cache_dir}")
            mmap = np.memmap(images_path, dtype=np.uint8, mode="w+", shape=tuple(expected_meta["shape"]))
            for index, record in enumerate(self.records_data):
                mmap[index] = self._load_image(record)
                if (index + 1) % 5000 == 0:
                    print(f"cached images: {index + 1}/{len(self.records_data)}")
            mmap.flush()
            records_path.write_text(json.dumps(expected_records, indent=2))
            meta_path.write_text(json.dumps(expected_meta, indent=2))

        return np.memmap(images_path, dtype=np.uint8, mode="r", shape=tuple(expected_meta["shape"]))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def get_angle(self, index: int) -> float:
        return float(self.records_data[self.sample_indices[index]]["user/angle"])

    def get_throttle(self, index: int) -> float:
        return float(self.records_data[self.sample_indices[index]]["user/throttle"])

    def get_all_angles(self, indices: list[int] | None = None) -> np.ndarray:
        if indices is None:
            indices = list(range(len(self)))
        return np.asarray([self.get_angle(index) for index in indices], dtype=np.float32)

    def get_all_throttles(self, indices: list[int] | None = None) -> np.ndarray:
        if indices is None:
            indices = list(range(len(self)))
        return np.asarray([self.get_throttle(index) for index in indices], dtype=np.float32)

    def __getitem__(self, index: int):
        record_index = self.sample_indices[index]
        record = self.records_data[record_index]
        if self.image_memmap is not None:
            image = np.array(self.image_memmap[record_index], copy=True)
        elif self.image_cache is None:
            image = self._load_image(record)
        else:
            image = self.image_cache[record_index]

        angle = float(record["user/angle"])
        throttle = float(record["user/throttle"])
        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            image = image[:, :, ::-1].copy()
            angle = -angle

        steer_class = closest_bin(angle, self.steering_centers)
        throttle_class = closest_bin(throttle, self.throttle_centers)
        return (
            torch.from_numpy(image),
            torch.tensor(steer_class, dtype=torch.long),
            torch.tensor(throttle_class, dtype=torch.long),
            torch.tensor(angle, dtype=torch.float32),
            torch.tensor(throttle, dtype=torch.float32),
        )


class OfficialCategoricalModel(nn.Module):
    def __init__(self, steering_bins: int = 15, throttle_bins: int = 20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 13, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.steer_head = nn.Linear(50, steering_bins)
        self.throttle_head = nn.Linear(50, throttle_bins)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x[:, :, 10:, :]
        x = x.float() / 255.0
        features = self.features(x)
        trunk = self.trunk(features)
        return {
            "steer_logits": self.steer_head(trunk),
            "throttle_logits": self.throttle_head(trunk),
        }


def init_from_regression_checkpoint(model: OfficialCategoricalModel, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    key_map = {}
    for key in source_state:
        if key.startswith("features."):
            key_map[key] = key
        elif key.startswith("head.1."):
            key_map[key] = key.replace("head.1.", "trunk.1.")
        elif key.startswith("head.4."):
            key_map[key] = key.replace("head.4.", "trunk.4.")

    copied = []
    skipped = []
    for source_key, target_key in key_map.items():
        if target_key in target_state and target_state[target_key].shape == source_state[source_key].shape:
            target_state[target_key].copy_(source_state[source_key])
            copied.append(f"{source_key}->{target_key}")
        else:
            skipped.append(f"{source_key}->{target_key}")
    model.load_state_dict(target_state)
    print(f"initialized from regression checkpoint: {checkpoint_path}")
    print(f"copied regression tensors: {len(copied)}")
    if skipped:
        print(f"skipped regression tensors: {skipped}")
    if len(copied) < len(key_map):
        raise RuntimeError(f"only initialized {len(copied)}/{len(key_map)} tensors from regression checkpoint")


def official_categorical_loss(
    outputs: dict[str, torch.Tensor],
    steer_classes: torch.Tensor,
    throttle_classes: torch.Tensor,
    throttle_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    steer_ce = F.cross_entropy(outputs["steer_logits"], steer_classes)
    throttle_ce = F.cross_entropy(outputs["throttle_logits"], throttle_classes)
    loss = steer_ce + throttle_loss_weight * throttle_ce
    with torch.no_grad():
        steer_acc = (outputs["steer_logits"].argmax(dim=-1) == steer_classes).float().mean()
        throttle_acc = (outputs["throttle_logits"].argmax(dim=-1) == throttle_classes).float().mean()
    return loss, {
        "loss": float(loss.item()),
        "steer_ce": float(steer_ce.item()),
        "throttle_ce": float(throttle_ce.item()),
        "steer_top1_acc": float(steer_acc.item()),
        "throttle_top1_acc": float(throttle_acc.item()),
    }


def run_epoch(model, loader, optimizer, device, throttle_loss_weight: float, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "steer_ce": 0.0, "throttle_ce": 0.0, "steer_top1_acc": 0.0, "throttle_top1_acc": 0.0}
    total_items = 0
    for images, steer_classes, throttle_classes, _angles, _throttles in loader:
        images = images.to(device, non_blocking=True)
        steer_classes = steer_classes.to(device, non_blocking=True)
        throttle_classes = throttle_classes.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss, metrics = official_categorical_loss(outputs, steer_classes, throttle_classes, throttle_loss_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_items += batch_size
    return {key: value / max(1, total_items) for key, value in totals.items()}


def print_distribution(name: str, values: np.ndarray, centers: np.ndarray) -> None:
    counts = np.zeros(len(centers), dtype=np.int64)
    for value in values:
        counts[closest_bin(float(value), centers)] += 1
    print(f"{name}_counts: {counts.tolist()}")


def print_value_diagnostics(name: str, values: np.ndarray) -> None:
    abs_values = np.abs(values)
    print(
        f"{name}: n={len(values)} min={values.min():+.4f} max={values.max():+.4f} "
        f"mean={values.mean():+.4f} std={values.std():.4f} "
        f"abs_p50={np.percentile(abs_values, 50):.4f} abs_p95={np.percentile(abs_values, 95):.4f} "
        f"abs_p99={np.percentile(abs_values, 99):.4f}"
    )


def make_steer_balanced_sampler(
    dataset: OfficialCategoricalDonkeyDataset,
    indices: list[int],
    centers: np.ndarray,
    max_weight: float,
    flip_prob: float,
) -> WeightedRandomSampler:
    counts = np.zeros(len(centers), dtype=np.float64)
    sample_classes = []
    for index in indices:
        angle = dataset.get_angle(index)
        cls = closest_bin(angle, centers)
        flip_cls = closest_bin(-angle, centers)
        counts[cls] += 1.0 - flip_prob
        counts[flip_cls] += flip_prob
        sample_classes.append((cls, flip_cls))
    safe_counts = np.maximum(counts, 1.0)
    weights_by_class = np.clip(np.sqrt(safe_counts.sum() / safe_counts), 1.0, max_weight)
    sample_weights = torch.tensor(
        [
            (1.0 - flip_prob) * weights_by_class[cls] + flip_prob * weights_by_class[flip_cls]
            for cls, flip_cls in sample_classes
        ],
        dtype=torch.double,
    )
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def collect_predictions(
    model,
    loader,
    device,
    steering_centers: np.ndarray,
    throttle_centers: np.ndarray,
) -> dict[str, float | list[float] | list[int]]:
    model.eval()
    pred_angles = []
    true_angles = []
    pred_throttles = []
    true_throttles = []
    pred_steer_classes = []
    true_steer_classes = []
    pred_throttle_classes = []
    true_throttle_classes = []
    steer_centers_tensor = torch.tensor(steering_centers, dtype=torch.float32, device=device)
    throttle_centers_tensor = torch.tensor(throttle_centers, dtype=torch.float32, device=device)
    with torch.no_grad():
        for images, steer_classes, throttle_classes, angles, throttles in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            steer_pred_cls = outputs["steer_logits"].argmax(dim=-1)
            throttle_pred_cls = outputs["throttle_logits"].argmax(dim=-1)
            pred_angles.append(steer_centers_tensor[steer_pred_cls].cpu().numpy())
            true_angles.append(angles.numpy())
            pred_throttles.append(throttle_centers_tensor[throttle_pred_cls].cpu().numpy())
            true_throttles.append(throttles.numpy())
            pred_steer_classes.append(steer_pred_cls.cpu().numpy())
            true_steer_classes.append(steer_classes.numpy())
            pred_throttle_classes.append(throttle_pred_cls.cpu().numpy())
            true_throttle_classes.append(throttle_classes.numpy())

    pred_angles_arr = np.concatenate(pred_angles)
    true_angles_arr = np.concatenate(true_angles)
    pred_throttles_arr = np.concatenate(pred_throttles)
    true_throttles_arr = np.concatenate(true_throttles)
    pred_steer_cls_arr = np.concatenate(pred_steer_classes)
    true_steer_cls_arr = np.concatenate(true_steer_classes)
    pred_throttle_cls_arr = np.concatenate(pred_throttle_classes)
    true_throttle_cls_arr = np.concatenate(true_throttle_classes)
    return {
        "steer_mae": float(np.mean(np.abs(pred_angles_arr - true_angles_arr))),
        "steer_rmse": float(np.sqrt(np.mean((pred_angles_arr - true_angles_arr) ** 2))),
        "throttle_mae": float(np.mean(np.abs(pred_throttles_arr - true_throttles_arr))),
        "throttle_rmse": float(np.sqrt(np.mean((pred_throttles_arr - true_throttles_arr) ** 2))),
        "pred_abs_percentiles": np.percentile(np.abs(pred_angles_arr), [50, 90, 95, 99, 100]).tolist(),
        "true_abs_percentiles": np.percentile(np.abs(true_angles_arr), [50, 90, 95, 99, 100]).tolist(),
        "pred_steer_class_counts": np.bincount(pred_steer_cls_arr, minlength=len(steering_centers)).tolist(),
        "true_steer_class_counts": np.bincount(true_steer_cls_arr, minlength=len(steering_centers)).tolist(),
        "pred_throttle_class_counts": np.bincount(pred_throttle_cls_arr, minlength=len(throttle_centers)).tolist(),
        "true_throttle_class_counts": np.bincount(true_throttle_cls_arr, minlength=len(throttle_centers)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-dir", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-head", type=int, default=0)
    parser.add_argument("--drop-tail", type=int, default=0)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--memmap-cache-dir", type=Path)
    parser.add_argument("--min-throttle", type=float)
    parser.add_argument("--max-abs-angle", type=float)
    parser.add_argument("--steering-bins", type=int, default=15)
    parser.add_argument("--steering-min", type=float, default=-1.0)
    parser.add_argument("--steering-max", type=float, default=1.0)
    parser.add_argument("--throttle-bins", type=int, default=20)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--sampler", choices=["none", "steer-balanced"], default="none")
    parser.add_argument("--sampler-weight-max", type=float, default=6.0)
    parser.add_argument("--throttle-loss-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from-regression", type=Path)
    args = parser.parse_args()
    if args.resume and args.init_from_regression:
        parser.error("--resume and --init-from-regression cannot be used together")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        data_dir=[str(path) for path in args.data_dir],
        val_dir=[str(path) for path in args.val_dir] if args.val_dir else None,
        output_dir=str(args.output_dir),
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        cache_images=args.cache_images,
        memmap_cache_dir=str(args.memmap_cache_dir) if args.memmap_cache_dir else None,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
        steering_bins=args.steering_bins,
        steering_min=args.steering_min,
        steering_max=args.steering_max,
        throttle_bins=args.throttle_bins,
        throttle_min=args.throttle_min,
        throttle_max=args.throttle_max,
        flip_prob=args.flip_prob,
        sampler=args.sampler,
        sampler_weight_max=args.sampler_weight_max,
        throttle_loss_weight=args.throttle_loss_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        init_from_regression=str(args.init_from_regression) if args.init_from_regression else None,
    )
    (args.output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    steering_centers = make_bin_centers(args.steering_bins, args.steering_min, args.steering_max)
    throttle_centers = make_bin_centers(args.throttle_bins, args.throttle_min, args.throttle_max)

    train_source_dataset = OfficialCategoricalDonkeyDataset(
        args.data_dir,
        steering_centers=steering_centers,
        throttle_centers=throttle_centers,
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        cache_images=args.cache_images,
        memmap_cache_dir=args.memmap_cache_dir,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
        flip_prob=args.flip_prob,
    )
    val_source_dataset = OfficialCategoricalDonkeyDataset(
        args.data_dir,
        steering_centers=steering_centers,
        throttle_centers=throttle_centers,
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        cache_images=args.cache_images,
        memmap_cache_dir=args.memmap_cache_dir,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
        flip_prob=0.0,
    )

    if args.val_dir:
        train_dataset = train_source_dataset
        train_indices = list(range(len(train_source_dataset)))
        val_dataset = OfficialCategoricalDonkeyDataset(
            args.val_dir,
            steering_centers=steering_centers,
            throttle_centers=throttle_centers,
            drop_head=args.drop_head,
            drop_tail=args.drop_tail,
            cache_images=args.cache_images,
            memmap_cache_dir=args.memmap_cache_dir,
            min_throttle=args.min_throttle,
            max_abs_angle=args.max_abs_angle,
            flip_prob=0.0,
        )
    else:
        train_indices, val_indices = split_indices(len(train_source_dataset), args.val_split, args.seed)
        train_dataset = Subset(train_source_dataset, train_indices)
        val_dataset = Subset(val_source_dataset, val_indices)

    sampler = None
    if args.sampler == "steer-balanced":
        sampler = make_steer_balanced_sampler(
            train_source_dataset,
            train_indices,
            steering_centers,
            args.sampler_weight_max,
            args.flip_prob,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OfficialCategoricalModel(args.steering_bins, args.throttle_bins).to(device)
    if args.init_from_regression:
        init_from_regression_checkpoint(model, args.init_from_regression, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        if resume_checkpoint.get("head_type") != "official_categorical_v1":
            raise ValueError(f"unsupported checkpoint head_type: {resume_checkpoint.get('head_type')}")
        checkpoint_steer = np.asarray(resume_checkpoint["steering_centers"], dtype=np.float32)
        checkpoint_throttle = np.asarray(resume_checkpoint["throttle_centers"], dtype=np.float32)
        if not np.allclose(checkpoint_steer, steering_centers, atol=1e-5):
            raise ValueError("resume checkpoint steering centers do not match current configuration")
        if not np.allclose(checkpoint_throttle, throttle_centers, atol=1e-5):
            raise ValueError("resume checkpoint throttle centers do not match current configuration")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val = float(resume_checkpoint.get("best_val", resume_checkpoint.get("val_metrics", {}).get("loss", best_val)))
        best_epoch = int(resume_checkpoint.get("best_epoch", resume_checkpoint.get("epoch", 0)))
        stale_epochs = int(resume_checkpoint.get("stale_epochs", 0))
        history = resume_checkpoint.get("history", [])
        if not history and (args.output_dir / "history.json").exists():
            history = json.loads((args.output_dir / "history.json").read_text())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    all_angles = train_source_dataset.get_all_angles()
    train_angles = train_source_dataset.get_all_angles(train_indices)
    all_throttles = train_source_dataset.get_all_throttles()
    train_throttles = train_source_dataset.get_all_throttles(train_indices)
    print(f"device: {device}")
    print(f"samples train_source={len(train_source_dataset)} train={len(train_dataset)} val={len(val_dataset)}")
    print(f"filtered samples: {train_source_dataset.filtered_count}")
    print(f"parameters: {count_parameters(model):,}")
    print_value_diagnostics("all angles", all_angles)
    print_value_diagnostics("train angles", train_angles)
    print_value_diagnostics("all throttles", all_throttles)
    print_value_diagnostics("train throttles", train_throttles)
    print(f"steering_centers: {steering_centers.tolist()}")
    print(f"throttle_centers: {throttle_centers.tolist()}")
    print_distribution("steering", train_angles, steering_centers)
    print_distribution("throttle", train_throttles, throttle_centers)
    for summary in train_source_dataset.tub_summaries:
        print(f"road_stats {summary}")
    print(model)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.throttle_loss_weight, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args.throttle_loss_weight, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_steer_ce={val_metrics['steer_ce']:.6f} "
            f"val_thr_ce={val_metrics['throttle_ce']:.6f} val_steer_acc={val_metrics['steer_top1_acc']:.4f} "
            f"val_thr_acc={val_metrics['throttle_top1_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "head_type": "official_categorical_v1",
                "steering_centers": steering_centers.tolist(),
                "throttle_centers": throttle_centers.tolist(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale_epochs += 1

        last_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "head_type": "official_categorical_v1",
            "steering_centers": steering_centers.tolist(),
            "throttle_centers": throttle_centers.tolist(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
        }
        torch.save(last_checkpoint, args.output_dir / "last.pt")
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch} val_loss={best_val:.6f}")
            break

    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    diagnostics = collect_predictions(model, val_loader, device, steering_centers, throttle_centers)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"best_epoch={best_epoch} best_val_loss={best_val:.6f}")
    print(f"diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()

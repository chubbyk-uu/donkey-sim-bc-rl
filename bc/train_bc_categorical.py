import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from train_bc import count_parameters, set_seed, split_indices


@dataclass
class TrainConfig:
    data_dir: list[str]
    val_dir: list[str] | None
    output_dir: str
    drop_head: int
    drop_tail: int
    cache_images: bool
    min_throttle: float | None
    max_abs_angle: float | None
    num_bins: int
    steering_min: float
    steering_max: float
    flip_prob: float
    class_weight_min: float
    class_weight_max: float
    throttle_loss_weight: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    val_split: float
    seed: int
    init_from_regression: str | None


def make_bins(num_bins: int, steering_min: float, steering_max: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(steering_min, steering_max, num_bins + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def encode_soft_steering(angle: float, centers: np.ndarray) -> np.ndarray:
    label = np.zeros(len(centers), dtype=np.float32)
    if len(centers) < 2:
        label[0] = 1.0
        return label

    bin_width = float(centers[1] - centers[0])
    clipped = float(np.clip(angle, centers[0], centers[-1]))
    pos = (clipped - float(centers[0])) / bin_width
    lo = int(np.floor(pos))
    lo = max(0, min(lo, len(centers) - 1))
    hi = min(lo + 1, len(centers) - 1)
    if hi == lo:
        label[lo] = 1.0
        return label

    w_hi = float(pos - lo)
    w_lo = 1.0 - w_hi
    label[lo] = w_lo
    label[hi] = w_hi
    return label


def decode_soft_steering(label: np.ndarray, centers: np.ndarray) -> float:
    return float(np.sum(label * centers))


def closest_bin(angle: float, centers: np.ndarray) -> int:
    clipped = float(np.clip(angle, centers[0], centers[-1]))
    return int(np.argmin(np.abs(centers - clipped)))


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class CategoricalDonkeyDataset(Dataset):
    def __init__(
        self,
        data_dirs: list[Path],
        centers: np.ndarray,
        drop_head: int = 0,
        drop_tail: int = 0,
        cache_images: bool = False,
        min_throttle: float | None = None,
        max_abs_angle: float | None = None,
        flip_prob: float = 0.0,
    ):
        self.centers = centers
        self.flip_prob = flip_prob
        self.records_data: list[dict] = []
        self.sample_indices: list[int] = []
        self.tub_summaries: list[dict] = []
        self.filtered_count = 0
        self.cache_images = cache_images

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
                    "throttle_mean": float(throttle_arr.mean()),
                }
            )

        if not self.sample_indices:
            raise ValueError("no samples selected")

        self.image_cache = self._load_image_cache() if cache_images else None

    def _load_image_cache(self) -> list[np.ndarray]:
        return [self._load_image(record) for record in self.records_data]

    def _load_image(self, record: dict) -> np.ndarray:
        image_path = Path(record["_data_dir"]) / record["cam/image_array"]
        image = Image.open(image_path).convert("RGB")
        image_arr = np.array(image, dtype=np.uint8, copy=True)
        return np.transpose(image_arr, (2, 0, 1))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def get_angle(self, index: int) -> float:
        return float(self.records_data[self.sample_indices[index]]["user/angle"])

    def get_all_angles(self, indices: list[int] | None = None) -> np.ndarray:
        if indices is None:
            indices = list(range(len(self)))
        return np.asarray([self.get_angle(index) for index in indices], dtype=np.float32)

    def __getitem__(self, index: int):
        record_index = self.sample_indices[index]
        record = self.records_data[record_index]
        if self.image_cache is None:
            image = self._load_image(record)
        else:
            image = self.image_cache[record_index]

        angle = float(record["user/angle"])
        throttle = float(record["user/throttle"])
        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            image = image[:, :, ::-1].copy()
            angle = -angle

        steer_label = encode_soft_steering(angle, self.centers)
        return (
            torch.from_numpy(image),
            torch.from_numpy(steer_label),
            torch.tensor([throttle], dtype=torch.float32),
            torch.tensor(angle, dtype=torch.float32),
        )


class CategoricalSteeringModel(nn.Module):
    def __init__(self, num_bins: int = 21):
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
        self.steer_head = nn.Linear(50, num_bins)
        self.throttle_head = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x[:, :, 10:, :]
        x = x.float() / 255.0
        features = self.features(x)
        trunk = self.trunk(features)
        return {
            "steer_logits": self.steer_head(trunk),
            "throttle_pred": self.throttle_head(trunk),
        }


def init_from_regression_checkpoint(model: CategoricalSteeringModel, checkpoint_path: Path, device: torch.device) -> None:
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


def categorical_loss(
    outputs: dict[str, torch.Tensor],
    steer_labels: torch.Tensor,
    throttles: torch.Tensor,
    class_weights: torch.Tensor,
    throttle_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = torch.log_softmax(outputs["steer_logits"], dim=-1)
    steer_ce = -(steer_labels * log_probs).sum(dim=-1)
    sample_weights = (steer_labels * class_weights).sum(dim=-1)
    weighted_steer_ce = steer_ce * sample_weights
    throttle_mse = (outputs["throttle_pred"] - throttles).pow(2).mean(dim=-1)
    loss = weighted_steer_ce.mean() + throttle_loss_weight * throttle_mse.mean()

    with torch.no_grad():
        pred_class = outputs["steer_logits"].argmax(dim=-1)
        true_class = steer_labels.argmax(dim=-1)
        top1_acc = (pred_class == true_class).float().mean()
    metrics = {
        "loss": float(loss.item()),
        "steer_ce": float(weighted_steer_ce.mean().item()),
        "throttle_mse": float(throttle_mse.mean().item()),
        "steer_top1_acc": float(top1_acc.item()),
    }
    return loss, metrics


def run_epoch(model, loader, optimizer, device, class_weights, throttle_loss_weight: float, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "steer_ce": 0.0, "throttle_mse": 0.0, "steer_top1_acc": 0.0}
    total_items = 0

    for images, steer_labels, throttles, _angles in loader:
        images = images.to(device, non_blocking=True)
        steer_labels = steer_labels.to(device, non_blocking=True)
        throttles = throttles.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss, metrics = categorical_loss(outputs, steer_labels, throttles, class_weights, throttle_loss_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_items += batch_size

    return {key: value / max(1, total_items) for key, value in totals.items()}


def compute_class_weights(
    dataset: CategoricalDonkeyDataset,
    indices: list[int],
    centers: np.ndarray,
    min_weight: float,
    max_weight: float,
    flip_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(len(centers), dtype=np.float64)
    for index in indices:
        angle = dataset.get_angle(index)
        counts[closest_bin(angle, centers)] += 1.0 - flip_prob
        counts[closest_bin(-angle, centers)] += flip_prob
    safe_counts = np.maximum(counts, 1.0)
    total = safe_counts.sum()
    raw = np.sqrt(total / safe_counts)
    weights = np.clip(raw, min_weight, max_weight).astype(np.float32)
    return counts.astype(np.float32), weights


def print_angle_diagnostics(name: str, angles: np.ndarray) -> None:
    abs_angles = np.abs(angles)
    print(
        f"{name}: n={len(angles)} min={angles.min():+.4f} max={angles.max():+.4f} "
        f"mean={angles.mean():+.4f} std={angles.std():.4f} "
        f"abs_p50={np.percentile(abs_angles, 50):.4f} abs_p95={np.percentile(abs_angles, 95):.4f} "
        f"abs_p99={np.percentile(abs_angles, 99):.4f}"
    )


def collect_predictions(model, loader, device, centers: np.ndarray) -> dict[str, float | list[float]]:
    model.eval()
    pred_angles = []
    true_angles = []
    pred_throttles = []
    true_throttles = []
    centers_tensor = torch.tensor(centers, dtype=torch.float32, device=device)
    with torch.no_grad():
        for images, _labels, throttles, angles in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs["steer_logits"], dim=-1)
            decoded = (probs * centers_tensor).sum(dim=-1)
            pred_angles.append(decoded.cpu().numpy())
            true_angles.append(angles.numpy())
            pred_throttles.append(outputs["throttle_pred"].squeeze(-1).cpu().numpy())
            true_throttles.append(throttles.squeeze(-1).numpy())

    pred_angles_arr = np.concatenate(pred_angles)
    true_angles_arr = np.concatenate(true_angles)
    pred_throttles_arr = np.concatenate(pred_throttles)
    true_throttles_arr = np.concatenate(true_throttles)
    return {
        "steer_mae": float(np.mean(np.abs(pred_angles_arr - true_angles_arr))),
        "steer_rmse": float(np.sqrt(np.mean((pred_angles_arr - true_angles_arr) ** 2))),
        "pred_abs_percentiles": np.percentile(np.abs(pred_angles_arr), [50, 90, 95, 99, 100]).tolist(),
        "true_abs_percentiles": np.percentile(np.abs(true_angles_arr), [50, 90, 95, 99, 100]).tolist(),
        "throttle_rmse": float(np.sqrt(np.mean((pred_throttles_arr - true_throttles_arr) ** 2))),
    }


def verify_soft_label_round_trip(centers: np.ndarray) -> None:
    for angle in [-0.5, -0.05, 0.0, 0.05, 0.3, 0.65]:
        label = encode_soft_steering(angle, centers)
        decoded = decode_soft_steering(label, centers)
        clipped = float(np.clip(angle, centers[0], centers[-1]))
        if abs(clipped - decoded) > 1e-5:
            raise AssertionError(f"soft label round-trip failed: angle={angle} decoded={decoded}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-dir", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-head", type=int, default=0)
    parser.add_argument("--drop-tail", type=int, default=0)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--min-throttle", type=float)
    parser.add_argument("--max-abs-angle", type=float)
    parser.add_argument("--num-bins", type=int, default=21)
    parser.add_argument("--steering-min", type=float, default=-0.7)
    parser.add_argument("--steering-max", type=float, default=0.7)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--class-weight-min", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=8.0)
    parser.add_argument("--throttle-loss-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
        num_bins=args.num_bins,
        steering_min=args.steering_min,
        steering_max=args.steering_max,
        flip_prob=args.flip_prob,
        class_weight_min=args.class_weight_min,
        class_weight_max=args.class_weight_max,
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

    bin_edges, bin_centers = make_bins(args.num_bins, args.steering_min, args.steering_max)
    verify_soft_label_round_trip(bin_centers)

    train_source_dataset = CategoricalDonkeyDataset(
        args.data_dir,
        centers=bin_centers,
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        cache_images=args.cache_images,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
        flip_prob=args.flip_prob,
    )
    val_source_dataset = CategoricalDonkeyDataset(
        args.data_dir,
        centers=bin_centers,
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        cache_images=args.cache_images,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
        flip_prob=0.0,
    )

    if args.val_dir:
        train_dataset = train_source_dataset
        train_indices = list(range(len(train_source_dataset)))
        val_dataset = CategoricalDonkeyDataset(
            args.val_dir,
            centers=bin_centers,
            drop_head=args.drop_head,
            drop_tail=args.drop_tail,
            cache_images=args.cache_images,
            min_throttle=args.min_throttle,
            max_abs_angle=args.max_abs_angle,
            flip_prob=0.0,
        )
    else:
        train_indices, val_indices = split_indices(len(train_source_dataset), args.val_split, args.seed)
        train_dataset = Subset(train_source_dataset, train_indices)
        val_dataset = Subset(val_source_dataset, val_indices)

    class_counts, class_weights_np = compute_class_weights(
        train_source_dataset,
        train_indices,
        bin_centers,
        args.class_weight_min,
        args.class_weight_max,
        args.flip_prob,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CategoricalSteeringModel(num_bins=args.num_bins).to(device)
    if args.init_from_regression:
        init_from_regression_checkpoint(model, args.init_from_regression, device)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        if resume_checkpoint.get("head_type") != "categorical_steer_v1":
            raise ValueError(f"unsupported checkpoint head_type: {resume_checkpoint.get('head_type')}")
        checkpoint_centers = np.asarray(resume_checkpoint["bin_centers"], dtype=np.float32)
        if not np.allclose(checkpoint_centers, bin_centers, atol=1e-5):
            raise ValueError("resume checkpoint bin centers do not match current bin configuration")
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
        shuffle=True,
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
    print(f"device: {device}")
    print(f"samples train_source={len(train_source_dataset)} train={len(train_dataset)} val={len(val_dataset)}")
    print(f"filtered samples: {train_source_dataset.filtered_count}")
    print(f"parameters: {count_parameters(model):,}")
    print_angle_diagnostics("all angles", all_angles)
    print_angle_diagnostics("train angles", train_angles)
    for summary in train_source_dataset.tub_summaries:
        print(f"road_stats {summary}")
    print(f"bin_centers: {bin_centers.tolist()}")
    print(f"class_counts_effective: {class_counts.tolist()}")
    print(f"class_weights: {class_weights_np.tolist()}")
    print(model)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            args.throttle_loss_weight,
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            class_weights,
            args.throttle_loss_weight,
            train=False,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_ce={val_metrics['steer_ce']:.6f} val_thr_mse={val_metrics['throttle_mse']:.6f} "
            f"val_acc={val_metrics['steer_top1_acc']:.4f}"
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
                "head_type": "categorical_steer_v1",
                "bin_edges": bin_edges.tolist(),
                "bin_centers": bin_centers.tolist(),
                "class_weights": class_weights_np.tolist(),
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
            "head_type": "categorical_steer_v1",
            "bin_edges": bin_edges.tolist(),
            "bin_centers": bin_centers.tolist(),
            "class_weights": class_weights_np.tolist(),
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
    diagnostics = collect_predictions(model, val_loader, device, bin_centers)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"best_epoch={best_epoch} best_val_loss={best_val:.6f}")
    print(f"diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()

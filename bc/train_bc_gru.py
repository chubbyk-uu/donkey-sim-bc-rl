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

from train_bc import set_seed, split_indices


@dataclass
class TrainConfig:
    data_dir: list[str]
    val_dir: list[str] | None
    output_dir: str
    drop_head: int
    drop_tail: int
    sequence_length: int
    frame_stride: int
    feature_dim: int
    hidden_size: int
    num_layers: int
    batch_size: int
    epochs: int
    patience: int
    cnn_learning_rate: float
    rnn_learning_rate: float
    weight_decay: float
    val_split: float
    seed: int
    min_throttle: float | None
    max_abs_angle: float | None


class DonkeySequenceDataset(Dataset):
    def __init__(
        self,
        data_dirs: list[Path],
        drop_head: int = 0,
        drop_tail: int = 0,
        sequence_length: int = 8,
        frame_stride: int = 1,
        min_throttle: float | None = None,
        max_abs_angle: float | None = None,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.sequence_span = (sequence_length - 1) * frame_stride
        self.records_data: list[dict] = []
        self.sample_indices: list[int] = []
        self.tub_summaries: list[dict] = []
        self.filtered_count = 0

        for data_dir in data_dirs:
            records = sorted(data_dir.glob("record_*.json"), key=lambda p: int(p.stem.split("_")[1]))
            end = len(records) - drop_tail if drop_tail else len(records)
            records = records[drop_head:end]
            if len(records) <= self.sequence_span:
                raise ValueError(f"not enough records selected in {data_dir}")

            tub_start = len(self.records_data)
            tub_records = [json.loads(record_path.read_text()) for record_path in records]
            samples_before = len(self.sample_indices)

            for record in tub_records:
                record["_data_dir"] = str(data_dir)
                self.records_data.append(record)

            for local_index, record in enumerate(tub_records):
                if local_index < self.sequence_span:
                    continue
                angle = float(record["user/angle"])
                throttle = float(record["user/throttle"])
                if min_throttle is not None and throttle < min_throttle:
                    self.filtered_count += 1
                    continue
                if max_abs_angle is not None and abs(angle) > max_abs_angle:
                    self.filtered_count += 1
                    continue
                self.sample_indices.append(tub_start + local_index)

            self.tub_summaries.append(
                {
                    "data_dir": str(data_dir),
                    "records": len(records),
                    "samples": len(self.sample_indices) - samples_before,
                }
            )

        if not self.sample_indices:
            raise ValueError("no sequence samples selected")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _load_frame(self, record: dict) -> np.ndarray:
        image_path = Path(record["_data_dir"]) / record["cam/image_array"]
        image = Image.open(image_path).convert("RGB")
        image_arr = np.asarray(image, dtype=np.uint8)
        return np.transpose(image_arr, (2, 0, 1))

    def __getitem__(self, index: int):
        record_index = self.sample_indices[index]
        start_index = record_index - self.sequence_span
        frames = []
        for frame_index in range(start_index, record_index + 1, self.frame_stride):
            frames.append(self._load_frame(self.records_data[frame_index]))

        record = self.records_data[record_index]
        action = np.array(
            [float(record["user/angle"]), float(record["user/throttle"])],
            dtype=np.float32,
        )
        sequence = np.stack(frames, axis=0)
        return torch.from_numpy(sequence), torch.from_numpy(action)


class CnnGruDonkeyModel(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 13, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.reshape(batch_size * sequence_length, channels, height, width)
        x = x[:, :, 10:, :]
        x = x.float() / 255.0
        features = self.cnn(x)
        features = features.reshape(batch_size, sequence_length, -1)
        _, hidden = self.gru(features)
        return self.head(hidden[-1])


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_items = 0

    for sequences, actions in loader:
        sequences = sequences.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            preds = model(sequences)
            loss = criterion(preds, actions)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = sequences.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(1, total_items)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-dir", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-head", type=int, default=0)
    parser.add_argument("--drop-tail", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rnn-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-throttle", type=float)
    parser.add_argument("--max-abs-angle", type=float)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=[str(path) for path in args.data_dir],
        val_dir=[str(path) for path in args.val_dir] if args.val_dir else None,
        output_dir=str(args.output_dir),
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        feature_dim=args.feature_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        cnn_learning_rate=args.cnn_learning_rate,
        rnn_learning_rate=args.rnn_learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
    )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    train_source_dataset = DonkeySequenceDataset(
        args.data_dir,
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        min_throttle=args.min_throttle,
        max_abs_angle=args.max_abs_angle,
    )
    if args.val_dir:
        train_dataset = train_source_dataset
        val_dataset = DonkeySequenceDataset(
            args.val_dir,
            drop_head=args.drop_head,
            drop_tail=args.drop_tail,
            sequence_length=args.sequence_length,
            frame_stride=args.frame_stride,
            min_throttle=args.min_throttle,
            max_abs_angle=args.max_abs_angle,
        )
    else:
        train_indices, val_indices = split_indices(len(train_source_dataset), args.val_split, args.seed)
        train_dataset = Subset(train_source_dataset, train_indices)
        val_dataset = Subset(train_source_dataset, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnGruDonkeyModel(
        feature_dim=args.feature_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.cnn.parameters(), "lr": args.cnn_learning_rate},
            {"params": list(model.gru.parameters()) + list(model.head.parameters()), "lr": args.rnn_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"device: {device}")
    print(
        f"samples train_source={len(train_source_dataset)} train={len(train_dataset)} val={len(val_dataset)} "
        f"sequence_length={args.sequence_length} stride={args.frame_stride}"
    )
    print(f"filtered samples: {train_source_dataset.filtered_count}")
    print(f"train tubs: {train_source_dataset.tub_summaries}")
    if args.val_dir:
        print(f"val tubs: {val_dataset.tub_summaries}")
    print(f"parameters: {count_parameters(model):,}")
    print(model)

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch} val_loss={best_val:.6f}")
                break

    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, args.output_dir / "last.pt")


if __name__ == "__main__":
    main()

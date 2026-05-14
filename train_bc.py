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


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    drop_head: int
    drop_tail: int
    history: int
    frame_stride: int
    cache_images: bool
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    val_split: float
    seed: int
    init_from_single_frame: str | None


class DonkeyTubDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        drop_head: int = 0,
        drop_tail: int = 0,
        history: int = 1,
        frame_stride: int = 1,
        cache_images: bool = False,
    ):
        if history < 1:
            raise ValueError("history must be >= 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.data_dir = data_dir
        records = sorted(data_dir.glob("record_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        end = len(records) - drop_tail if drop_tail else len(records)
        self.records = records[drop_head:end]
        self.records_data = [json.loads(record_path.read_text()) for record_path in self.records]
        self.history = history
        self.frame_stride = frame_stride
        self.history_span = (history - 1) * frame_stride
        if len(self.records) <= self.history_span:
            raise ValueError("no records selected")
        self.image_cache = self._load_image_cache() if cache_images else None

    def _load_image_cache(self) -> list[np.ndarray]:
        image_cache = []
        for record in self.records_data:
            image_cache.append(self._load_image(record))
        return image_cache

    def _load_image(self, record: dict) -> np.ndarray:
        image_path = self.data_dir / record["cam/image_array"]
        image = Image.open(image_path).convert("RGB")
        image_arr = np.asarray(image, dtype=np.uint8)
        return np.transpose(image_arr, (2, 0, 1))

    def __len__(self) -> int:
        return len(self.records) - self.history_span

    def __getitem__(self, index: int):
        record_index = index + self.history_span
        record = self.records_data[record_index]

        frames = []
        start_index = record_index - self.history_span
        for frame_index in range(start_index, record_index + 1, self.frame_stride):
            if self.image_cache is None:
                frame = self._load_image(self.records_data[frame_index])
            else:
                frame = self.image_cache[frame_index]
            frames.append(frame)

        action = np.array(
            [float(record["user/angle"]), float(record["user/throttle"])],
            dtype=np.float32,
        )
        image_stack = np.concatenate(frames, axis=0)
        return torch.from_numpy(image_stack), torch.from_numpy(action)


class NvidiaDonkeyModel(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2),
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 13, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(50, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, 10:, :]
        x = x.float() / 255.0
        x = self.features(x)
        return self.head(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_items = 0

    for images, actions in loader:
        images = images.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            preds = model(images)
            loss = criterion(preds, actions)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(1, total_items)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_from_single_frame_checkpoint(model: NvidiaDonkeyModel, checkpoint_path: Path, history: int, device: torch.device) -> None:
    if history < 2:
        raise ValueError("--init-from-single-frame is only useful when history > 1")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()

    first_weight_key = "features.0.weight"
    first_bias_key = "features.0.bias"
    target_state[first_weight_key].zero_()
    current_frame_start = (history - 1) * 3
    target_state[first_weight_key][:, current_frame_start : current_frame_start + 3, :, :] = source_state[first_weight_key]
    target_state[first_bias_key].copy_(source_state[first_bias_key])

    for key, value in source_state.items():
        if key in {first_weight_key, first_bias_key}:
            continue
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key].copy_(value)

    model.load_state_dict(target_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/d/WSL/log"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/bc_nvidia"))
    parser.add_argument("--drop-head", type=int, default=0)
    parser.add_argument("--drop-tail", type=int, default=100)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-from-single-frame", type=Path)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        history=args.history,
        frame_stride=args.frame_stride,
        cache_images=args.cache_images,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        init_from_single_frame=str(args.init_from_single_frame) if args.init_from_single_frame else None,
    )

    set_seed(cfg.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    dataset = DonkeyTubDataset(
        args.data_dir,
        drop_head=args.drop_head,
        drop_tail=args.drop_tail,
        history=args.history,
        frame_stride=args.frame_stride,
        cache_images=args.cache_images,
    )
    train_indices, val_indices = split_indices(len(dataset), args.val_split, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NvidiaDonkeyModel(input_channels=args.history * 3).to(device)
    if args.init_from_single_frame:
        init_from_single_frame_checkpoint(model, args.init_from_single_frame, args.history, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
        f"records used: {len(dataset)} train={len(train_dataset)} val={len(val_dataset)} "
        f"history={args.history} stride={args.frame_stride} cache_images={args.cache_images}"
    )
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

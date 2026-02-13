import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATA_ROOT = Path("datasets/RAF-DB")
OUTPUT_DIR = Path("emotion_recognition/models")
EPOCHS = 500
BATCH_SIZE = 256
IMAGE_SIZE = 112
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
LABEL_SMOOTHING = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


@dataclass
class EpochMetrics:
    loss: float
    acc: float


class RafDbEmotionDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = self._discover_classes()
        self.samples = self._gather_samples()

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def _discover_classes(self) -> Dict[str, int]:
        classes = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir() and entry.name.isdigit():
                classes.append(entry.name)
        if not classes:
            raise RuntimeError(f"No numeric class folders found under {self.root}")
        return {name: idx for idx, name in enumerate(classes)}

    def _gather_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMG_EXTENSIONS and image_path.is_file():
                    samples.append((image_path, class_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def class_counts(self) -> List[int]:
        counts = [0 for _ in range(self.num_classes)]
        for _, label in self.samples:
            counts[label] += 1
        return counts


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallEmotionCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 24),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(24, 48),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(48, 72),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(72, 96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    dataset_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Sequence[int]]:
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = RafDbEmotionDataset(dataset_root / "train", train_transform)
    valid_dataset = RafDbEmotionDataset(dataset_root / "valid", eval_transform)

    if train_dataset.num_classes != valid_dataset.num_classes:
        raise RuntimeError("Train/valid class count mismatch.")

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    return train_loader, valid_loader, train_dataset.class_counts()


def build_class_weights(class_counts: Sequence[int], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = torch.sqrt(counts.sum() / counts.clamp_min(1.0))
    weights = weights / weights.mean()
    return weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return EpochMetrics(
        loss=total_loss / total_samples,
        acc=total_correct / total_samples,
    )


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    current_acc: float,
    best_acc: float,
    config: Dict[str, object],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_acc": current_acc,
            "best_acc": best_acc,
            "config": config,
        },
        save_path,
    )
    metadata_path = save_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": save_path.name,
                "epoch": epoch,
                "current_acc": current_acc,
                "best_acc": best_acc,
                "config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_named_checkpoint(
    prefix: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    current_acc: float,
    best_acc: float,
    config: Dict[str, object],
) -> Path:
    filename = f"{prefix}_epoch{epoch:03d}_acc{current_acc * 100:.2f}.pth"
    save_path = OUTPUT_DIR / filename
    save_checkpoint(save_path, model, optimizer, epoch, current_acc, best_acc, config)
    return save_path


def main() -> None:
    seed_everything(SEED)
    config = {
        "data_root": str(DATA_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "label_smoothing": LABEL_SMOOTHING,
        "device": str(DEVICE),
    }

    print(f"Using torch {torch.__version__}")
    print(f"Using device: {DEVICE}")

    train_loader, valid_loader, class_counts = create_dataloaders(
        dataset_root=DATA_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    print(f"Class counts: {class_counts}")

    model = SmallEmotionCNN(num_classes=len(class_counts)).to(DEVICE)
    class_weights = build_class_weights(class_counts, DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Model params: {count_parameters(model):,}")

    best_acc = 0.0
    last_valid_acc = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, DEVICE, optimizer)
        valid_metrics = run_epoch(model, valid_loader, criterion, DEVICE)
        last_valid_acc = valid_metrics.acc
        scheduler.step()

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.acc:.4%} "
            f"valid_loss={valid_metrics.loss:.4f} "
            f"valid_acc={valid_metrics.acc:.4%} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if valid_metrics.acc >= best_acc:
            best_acc = valid_metrics.acc
            best_state_dict = copy.deepcopy(model.state_dict())
            named_best_path = save_named_checkpoint(
                "rafdb_tiny_cnn_best",
                model,
                optimizer,
                epoch,
                valid_metrics.acc,
                best_acc,
                config,
            )
            save_checkpoint(
                OUTPUT_DIR / "rafdb_tiny_cnn_best.pth",
                model,
                optimizer,
                epoch,
                valid_metrics.acc,
                best_acc,
                config,
            )
            print(
                f"Saved best checkpoint: epoch={epoch}, "
                f"valid_acc={valid_metrics.acc:.4%}, path={named_best_path}"
            )

    model.load_state_dict(best_state_dict)
    save_named_checkpoint(
        "rafdb_tiny_cnn_last",
        model,
        optimizer,
        EPOCHS,
        last_valid_acc,
        best_acc,
        config,
    )
    save_checkpoint(
        OUTPUT_DIR / "rafdb_tiny_cnn_last.pth",
        model,
        optimizer,
        EPOCHS,
        last_valid_acc,
        best_acc,
        config,
    )

    print(f"Best validation accuracy: {best_acc:.4%}")


if __name__ == "__main__":
    main()

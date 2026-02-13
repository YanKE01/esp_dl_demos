import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_recognition.train import RafDbEmotionDataset, SmallEmotionCNN, build_transforms


CHECKPOINT_PATH = Path("emotion_recognition/models/rafdb_tiny_cnn_last.pth")
DATA_ROOT = Path("datasets/RAF-DB")
IMAGE_SIZE = 96
BATCH_SIZE = 128
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EVAL_SAMPLES_PER_CLASS = 256


def load_checkpoint_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> SmallEmotionCNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SmallEmotionCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_loader(
    split_root: Path, image_size: int, batch_size: int, num_workers: int
) -> Tuple[DataLoader, RafDbEmotionDataset]:
    _, eval_transform = build_transforms(image_size)
    dataset = RafDbEmotionDataset(split_root, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


def build_subset_loader(
    split_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    samples_per_class: int,
) -> Tuple[DataLoader, RafDbEmotionDataset, int]:
    _, eval_transform = build_transforms(image_size)
    dataset = RafDbEmotionDataset(split_root, eval_transform)
    selected_indices = []
    class_buckets = {class_idx: 0 for class_idx in range(dataset.num_classes)}

    for index, (_, label) in enumerate(dataset.samples):
        if class_buckets[label] < samples_per_class:
            selected_indices.append(index)
            class_buckets[label] += 1

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset, len(selected_indices)


def evaluate(
    model: SmallEmotionCNN,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, List[int], List[int]]:
    total_correct = 0
    total_samples = 0
    per_class_correct = [0 for _ in range(num_classes)]
    per_class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for class_idx in range(num_classes):
                mask = labels == class_idx
                if mask.any():
                    per_class_total[class_idx] += mask.sum().item()
                    per_class_correct[class_idx] += (preds[mask] == labels[mask]).sum().item()

    accuracy = total_correct / total_samples
    return accuracy, per_class_correct, per_class_total


def print_metrics(split_name: str, dataset: RafDbEmotionDataset, accuracy: float, per_class_correct: List[int], per_class_total: List[int]) -> None:
    print(f"{split_name} samples: {len(dataset)}")
    print(f"{split_name} accuracy: {accuracy:.4%}")
    print(f"{split_name} per-class accuracy:")
    for class_name, class_idx in dataset.class_to_idx.items():
        class_acc = per_class_correct[class_idx] / max(per_class_total[class_idx], 1)
        print(
            f"  class {class_name}: "
            f"{per_class_correct[class_idx]}/{per_class_total[class_idx]} "
            f"({class_acc:.4%})"
        )


def main() -> None:
    train_loader, train_dataset, train_sample_count = build_subset_loader(
        split_root=DATA_ROOT / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        samples_per_class=TRAIN_EVAL_SAMPLES_PER_CLASS,
    )
    valid_loader, valid_dataset = build_loader(
        split_root=DATA_ROOT / "valid",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    model = load_checkpoint_model(CHECKPOINT_PATH, valid_dataset.num_classes, DEVICE)
    train_accuracy, train_per_class_correct, train_per_class_total = evaluate(
        model=model,
        loader=train_loader,
        device=DEVICE,
        num_classes=train_dataset.num_classes,
    )
    valid_accuracy, valid_per_class_correct, valid_per_class_total = evaluate(
        model=model,
        loader=valid_loader,
        device=DEVICE,
        num_classes=valid_dataset.num_classes,
    )

    print(f"Using torch {torch.__version__}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(
        f"Train eval subset: {train_sample_count}/{len(train_dataset)} "
        f"({TRAIN_EVAL_SAMPLES_PER_CLASS} per class)"
    )
    print_metrics("Train", train_dataset, train_accuracy, train_per_class_correct, train_per_class_total)
    print_metrics("Valid", valid_dataset, valid_accuracy, valid_per_class_correct, valid_per_class_total)


if __name__ == "__main__":
    main()

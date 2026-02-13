import sys
from pathlib import Path
from typing import Iterable, Tuple

import torch
from PIL import Image
from esp_ppq import QuantizationSetting, QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_recognition.train import (
    DATA_ROOT as TRAIN_DATA_ROOT,
    IMAGE_SIZE as TRAIN_IMAGE_SIZE,
    OUTPUT_DIR as TRAIN_OUTPUT_DIR,
    RafDbEmotionDataset,
    SmallEmotionCNN,
    build_transforms,
)


CHECKPOINT_PATH = TRAIN_OUTPUT_DIR / "rafdb_tiny_cnn_last.pth"
DATA_ROOT = TRAIN_DATA_ROOT
OUTPUT_PATH = TRAIN_OUTPUT_DIR / "emotion_cls.espdl"
TARGET = "esp32p4"
NUM_OF_BITS = 8
IMAGE_SIZE = TRAIN_IMAGE_SIZE
BATCH_SIZE = 32
CALIB_STEPS = 32
CALIB_SAMPLES = 512
NUM_WORKERS = 0
DEVICE = torch.device("cpu")
ENABLE_EQUALIZATION = True
TEST_IMAGE = DATA_ROOT / "valid" / "6" / "test_2391.jpg"


def build_calibration_loader(
    dataset_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> DataLoader:
    _, eval_transform = build_transforms(image_size)
    train_dataset = RafDbEmotionDataset(dataset_root / "train", eval_transform)

    if max_samples > 0:
        max_samples = min(max_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(max_samples)))

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        # Calibration data order must stay deterministic for quantization.
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_images,
    )


def collate_images(batch: Iterable[Tuple[torch.Tensor, int]]) -> torch.Tensor:
    return torch.stack([sample[0] for sample in batch], dim=0)


def move_batch_to_device(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


def build_quantization_setting(enable_equalization: bool) -> QuantizationSetting:
    setting = QuantizationSettingFactory.espdl_setting()
    if enable_equalization:
        setting.equalization = True
        setting.equalization_setting.iterations = 4
        setting.equalization_setting.value_threshold = 0.4
        setting.equalization_setting.opt_level = 2
        setting.equalization_setting.interested_layers = None
    return setting


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model = SmallEmotionCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def load_test_input(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    _, eval_transform = build_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0)
    return tensor.to(device)


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("config", {})


def main() -> None:
    checkpoint_config = load_checkpoint_config(CHECKPOINT_PATH)
    image_size = int(checkpoint_config.get("image_size", IMAGE_SIZE))
    data_root = Path(checkpoint_config.get("data_root", str(DATA_ROOT)))

    _, eval_transform = build_transforms(image_size)
    class_count = RafDbEmotionDataset(data_root / "train", eval_transform).num_classes

    model = load_model(CHECKPOINT_PATH, num_classes=class_count, device=DEVICE)
    calib_loader = build_calibration_loader(
        dataset_root=data_root,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_samples=CALIB_SAMPLES,
    )
    quant_setting = build_quantization_setting(ENABLE_EQUALIZATION)
    test_input = load_test_input(TEST_IMAGE, image_size, DEVICE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using torch {torch.__version__}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data root: {data_root}")
    print(f"Image size: {image_size}")
    print(f"Calibration batches: {len(calib_loader)}")
    print(f"Calibration steps: {CALIB_STEPS}")
    print(f"Target: {TARGET}, bits: {NUM_OF_BITS}, device: {DEVICE}")
    print(f"Test image: {TEST_IMAGE}")

    espdl_quantize_torch(
        model=model,
        espdl_export_file=OUTPUT_PATH.as_posix(),
        calib_dataloader=calib_loader,
        calib_steps=min(CALIB_STEPS, len(calib_loader)),
        input_shape=[1, 3, image_size, image_size],
        inputs=[test_input],
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=move_batch_to_device,
        setting=quant_setting,
        device=str(DEVICE),
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    print(f"Quantized model exported to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

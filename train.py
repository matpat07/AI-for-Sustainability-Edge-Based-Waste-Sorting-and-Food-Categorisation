#!/usr/bin/env python3
"""
train.py (Waste Classification Transfer Learning & ExecuTorch Export)

1. Loads a waste image dataset (TrashNet 6-class or Kaggle Waste Images 9-class).
2. Performs transfer learning with MobileNetV2 (fine-tuning top layers).
3. Evaluates accuracy on validation split.
4. Saves PyTorch weights and class labels (labels.txt).
5. Lowers and exports the fine-tuned model directly to ExecuTorch (.pte) with XNNPACK
   optimization for edge devices (e.g., Raspberry Pi).
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure site-packages takes precedence over the unbuilt local executorch/ repo
for p in list(sys.path):
    if "site-packages" in p:
        sys.path.remove(p)
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def find_default_dataset_dir() -> Path:
    """Finds available waste dataset directory in order of preference."""
    candidates = [
        Path("dataset/dataset-resized/dataset-resized"),
        Path("dataset/Waste Images"),
        Path("dataset/waste-images/Waste Images"),
        Path("dataset/dataset-resized"),
        Path("trashnet_raw/dataset-resized"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            subdirs = [
                d for d in candidate.iterdir()
                if d.is_dir() and not d.name.startswith((".", "__"))
            ]
            if subdirs:
                return candidate
    return candidates[0]


def get_data_loaders(data_dir: Path, batch_size: int = 32, val_split: float = 0.15):
    """Prepares training and validation DataLoaders with appropriate transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset twice to apply distinct train/val transforms without side effects
    train_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(data_dir), transform=val_transform)

    class_names = train_dataset.classes
    total_samples = len(train_dataset)

    # Deterministic reproducible split
    torch.manual_seed(42)
    indices = torch.randperm(total_samples).tolist()
    train_count = int((1.0 - val_split) * total_samples)

    train_ds = Subset(train_dataset, indices[:train_count])
    val_ds = Subset(val_dataset, indices[train_count:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Builds MobileNetV2 with pretrained weights and replaces the classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze early feature layers so training is fast
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the last two inverted residual blocks for higher fine-tuning accuracy
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Replace ImageNet classifier head with target waste classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


def train_model(model, train_loader, val_loader, epochs: int, lr: float, device: torch.device):
    """Trains the model and evaluates on the validation set."""
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Validation check
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = (correct / total * 100) if total > 0 else 0.0
        val_acc = (val_correct / val_total * 100) if val_total > 0 else 0.0
        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")


def export_to_executorch(model: nn.Module, output_path: Path):
    """Lowers the fine-tuned model to ExecuTorch with XNNPACK delegate."""
    print("[*] Lowering fine-tuned model to ExecuTorch with XNNPACK...")
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    model.cpu().eval()
    example_input = (torch.randn(1, 3, 224, 224),)
    exported_program = torch.export.export(model, example_input)

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )
    exec_prog = edge_program.to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(exec_prog.buffer)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Successfully exported ExecuTorch model to '{output_path}' ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Train waste classification model and export to ExecuTorch.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=find_default_dataset_dir(),
        help="Path to dataset directory (defaults to auto-detected waste dataset)",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("waste_classifier.pte"),
        help="Path for exported .pte ExecuTorch file",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("mobilenetv2_waste.pth"),
        help="Path for saving PyTorch weights (.pth)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"[!] Error: Dataset directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading dataset from: {data_dir}")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir, batch_size=args.batch_size
    )
    print(f"[+] Classes found ({len(class_names)}): {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Build model & transfer learning
    model = build_model(len(class_names), device)
    print(f"[*] Starting {args.epochs} epoch(s) of transfer learning...")
    train_model(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Save PyTorch weights
    model.cpu().eval()
    torch.save(model.state_dict(), args.weights)
    print(f"[+] Saved PyTorch weights to '{args.weights}'")

    # Save class names mapping for inference
    labels_file = args.output.parent / "labels.txt"
    with open(labels_file, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"[+] Saved class labels to '{labels_file}'")

    # Export to ExecuTorch with XNNPACK
    export_to_executorch(model, args.output)


if __name__ == "__main__":
    main()

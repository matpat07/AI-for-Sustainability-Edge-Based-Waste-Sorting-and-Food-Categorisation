#!/usr/bin/env python3
"""
train.py (Waste Classification Transfer Learning & ExecuTorch Export)

1. Prepares 4-class waste dataset (glass, biodegradable, plastic, metal).
2. Performs transfer learning with MobileNetV2 using Intel Core Ultra hardware acceleration.
3. Evaluates accuracy on held-out validation split.
4. Saves PyTorch weights (mobilenetv2_waste.pth) and class labels (labels.txt).
5. Exports the fine-tuned model directly to ExecuTorch (.pte) with XNNPACK optimization.
"""

import argparse
import json
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


def setup_4class_dataset() -> Path:
    """Sets up a clean 4-class waste dataset: glass, biodegradable, plastic, metal."""
    target_dir = Path("dataset/waste_4class")
    target_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "biodegradable": [
            "dataset/Waste Images/Organic_Waste",
            "dataset/waste-images/Waste Images/Organic_Waste",
        ],
        "glass": [
            "dataset/Waste Images/Glass",
            "dataset/waste-images/Waste Images/Glass",
            "dataset/dataset-resized/dataset-resized/glass",
        ],
        "metal": [
            "dataset/Waste Images/Aluminium",
            "dataset/waste-images/Waste Images/Aluminium",
            "dataset/dataset-resized/dataset-resized/metal",
        ],
        "plastic": [
            "dataset/Waste Images/Plastics",
            "dataset/waste-images/Waste Images/Plastics",
            "dataset/dataset-resized/dataset-resized/plastic",
        ],
    }

    for target_class, candidates in mapping.items():
        link_path = target_dir / target_class
        if not link_path.exists() and not link_path.is_symlink():
            for c in candidates:
                src_path = Path(c).resolve()
                if src_path.is_dir():
                    link_path.symlink_to(src_path)
                    break

    return target_dir


def detect_device() -> tuple[torch.device, str]:
    """Detects available hardware accelerator (Intel XPU, CUDA, or Intel Core Ultra CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA GPU: {torch.cuda.get_device_name(0)}"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), f"Intel XPU/Arc GPU: {torch.xpu.get_device_name(0)}"

    # Multi-threaded Intel Core Ultra CPU execution (AVX2 + VNNI)
    threads = min(os.cpu_count() or 4, 14)
    torch.set_num_threads(threads)
    return torch.device("cpu"), f"Intel Core Ultra CPU ({torch.get_num_threads()} threads, AVX2/oneDNN)"


def get_data_loaders(data_dir: Path, batch_size: int = 32, val_split: float = 0.15):
    """Prepares training and validation DataLoaders with data augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(data_dir), transform=val_transform)

    class_names = train_dataset.classes
    total_samples = len(train_dataset)

    # Deterministic reproducible 85% train / 15% validation split
    torch.manual_seed(42)
    indices = torch.randperm(total_samples).tolist()
    train_count = int((1.0 - val_split) * total_samples)

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    # Save held-out test split mapping for edge evaluation
    test_samples = []
    for idx in val_indices:
        full_path, class_idx = val_dataset.samples[idx]
        cls_name = class_names[class_idx]
        file_name = Path(full_path).name
        test_samples.append({"rel_path": f"{cls_name}/{file_name}", "class": cls_name})

    with open("test_split.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Builds MobileNetV2 with pretrained weights and fine-tuning head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze early feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze top residual blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Replace classifier head for the target classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


def train_model(model, train_loader, val_loader, epochs: int, lr: float, device: torch.device):
    """Fine-tunes the model with AdamW and CrossEntropyLoss."""
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()

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
        print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Train Acc: {train_acc:>5.2f}% | Val Acc: {val_acc:>5.2f}%")


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
    parser = argparse.ArgumentParser(description="Train 4-class waste classifier (glass, biodegradable, plastic, metal).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset directory (defaults to auto-generated 4-class dataset)",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
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

    # Prepare dataset
    if args.data_dir is None:
        data_dir = setup_4class_dataset()
    else:
        data_dir = args.data_dir

    if not data_dir.is_dir():
        print(f"[!] Error: Dataset directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading dataset from: {data_dir}")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir, batch_size=args.batch_size
    )
    print(f"[+] Classes ({len(class_names)}): {class_names}")

    device, device_desc = detect_device()
    print(f"[*] Hardware Accelerator: {device_desc}")

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

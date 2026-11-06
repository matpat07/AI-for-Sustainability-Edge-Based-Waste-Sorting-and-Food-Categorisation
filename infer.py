#!/usr/bin/env python3
"""
infer.py - Fast Single-Image Inference with ExecuTorch on Raspberry Pi
Usage:
    .venv/bin/python infer.py path/to/image.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import executorch.extension.pybindings.portable_lib as exec_lib
except ImportError as e:
    print(f"[!] Error loading ExecuTorch: {e}", file=sys.stderr)
    sys.exit(1)

MODEL_PATH = "models/waste_classifier.pte" if Path("models/waste_classifier.pte").exists() else "waste_classifier.pte"
LABELS_PATH = "models/labels.txt" if Path("models/labels.txt").exists() else "labels.txt"
RECYCLABLE_LABELS = {"cardboard", "glass", "metal", "paper", "plastic"}


def load_labels(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    parser = argparse.ArgumentParser(description="Fast Single-Image Inference with ExecuTorch on Raspberry Pi")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .pte model")
    parser.add_argument("--labels", default=LABELS_PATH, help="Path to labels.txt")
    parser.add_argument(
        "--cores",
        default="0,1",
        help="CPU cores to bind process to (e.g. '0,1' for 2 cores, 'all' for unconstrained)",
    )
    parser.add_argument("--nice", type=int, default=10, help="Process niceness (0-19)")
    args = parser.parse_args()

    # Apply core pinning and scheduling priority on Linux
    if hasattr(os, "sched_setaffinity") and args.cores and args.cores.lower() != "all":
        try:
            core_ids = {int(c.strip()) for c in args.cores.split(",")}
            os.sched_setaffinity(0, core_ids)
        except Exception:
            pass

    if hasattr(os, "nice") and args.nice is not None:
        try:
            os.nice(args.nice)
        except Exception:
            pass

    image_path = args.image
    if not Path(image_path).exists():
        print(f"Error: Image '{image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    labels = load_labels(args.labels)
    module = exec_lib._load_for_executorch(args.model)

    tensor = preprocess(image_path)

    t0 = time.perf_counter()
    outputs = module.run_method("forward", (tensor,))
    latency = (time.perf_counter() - t0) * 1000

    logits = np.array(outputs[0])[0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    pred_class = labels[pred_idx]
    confidence = probs[pred_idx] * 100
    category = "♻️  Recyclable (Dry Waste)" if pred_class in RECYCLABLE_LABELS else "🍂 Biodegradable (Wet Waste)"

    print("=" * 48)
    print(" ExecuTorch Edge Waste Classifier Result")
    print("=" * 48)
    print(f" Image:       {Path(image_path).name}")
    print(f" Prediction:  {pred_class.upper()}")
    print(f" Category:    {category}")
    print(f" Confidence:  {confidence:.2f}%")
    print(f" Latency:     {latency:.1f} ms")
    print("-" * 48)
    print(" Class Probabilities:")
    for name, p in zip(labels, probs):
        bar = "█" * int(p * 20)
        print(f"  {name:<14}: {p*100:>5.1f}% | {bar}")
    print("=" * 48)


if __name__ == "__main__":
    main()

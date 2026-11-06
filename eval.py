#!/usr/bin/env python3
"""
eval.py (ExecuTorch Waste Classifier Evaluation on Edge Device)

Evaluates waste_classifier.pte across test images or full dataset.
Calculates class accuracy, broad bin accuracy (recyclable vs organic),
inference latency, FPS, and peak memory usage on the Raspberry Pi.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

try:
    import executorch.extension.pybindings.portable_lib as exec_lib
except ImportError as e:
    print(f"[!] Error importing ExecuTorch portable_lib: {e}", file=sys.stderr)
    sys.exit(1)


# Map 6 classes to broad societal waste bins
RECYCLABLE_LABELS = {"cardboard", "glass", "metal", "paper", "plastic"}


def load_labels(path: str) -> list[str]:
    """Loads class names from file or defaults to TrashNet 6 classes."""
    if os.path.exists(path):
        with open(path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
            if labels:
                return labels
    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def preprocess(image_path: str) -> np.ndarray:
    """Preprocesses input image to normalized float32 tensor matching MobileNetV2."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


def get_ram_mb() -> float:
    """Returns current process Resident Set Size in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def collect_images(
    target_dir: str, labels: list[str], test_split_file: str = None
) -> list[tuple[str, str]]:
    """
    Collects images from target_dir. If test_split_file is provided,
    loads only the exact held-out split samples.
    """
    if test_split_file and os.path.exists(test_split_file):
        with open(test_split_file, "r") as f:
            items = json.load(f)
        collected = []
        for it in items:
            p = os.path.join(target_dir, it["rel_path"])
            if os.path.exists(p):
                collected.append((p, it["class"]))
            else:
                print(f"[!] Warning: test sample '{p}' not found on disk.")
        return collected

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    collected = []

    # Check if target_dir has class subdirectories
    subdirs = [
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith((".", "__"))
    ]

    has_class_subdirs = any(d in labels for d in subdirs)

    if has_class_subdirs:
        for root, dirs, files in os.walk(target_dir):
            parent = os.path.basename(root)
            if parent in labels:
                for f in sorted(files):
                    if f.lower().endswith(valid_exts) and not f.startswith("."):
                        collected.append((os.path.join(root, f), parent))
    else:
        # Flat directory of test images
        for f in sorted(os.listdir(target_dir)):
            if f.lower().endswith(valid_exts) and not f.startswith("."):
                collected.append((os.path.join(target_dir, f), "unknown"))

    return collected


def main():
    parser = argparse.ArgumentParser(description="Evaluate ExecuTorch compiled model on edge device.")
    parser.add_argument("--model", default="waste_classifier.pte", help="Path to .pte model")
    parser.add_argument("--labels", default="labels.txt", help="Path to labels.txt")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to images directory (defaults to dataset/dataset-resized/dataset-resized or test_images)",
    )
    parser.add_argument(
        "--test-split",
        default="test_split.json" if os.path.exists("test_split.json") else None,
        help="Path to test_split.json for strictly held-out evaluation",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to evaluate")
    args = parser.parse_args()

    # Determine data directory
    data_dir = args.data_dir
    if not data_dir:
        candidates = [
            "dataset/dataset-resized/dataset-resized",
            "dataset/dataset-resized",
            "test_images",
            "dataset/Waste Images",
        ]
        for c in candidates:
            if os.path.isdir(c):
                data_dir = c
                break

    if not data_dir or not os.path.isdir(data_dir):
        print(f"[!] Error: No valid image directory found. Specify with --data-dir.", file=sys.stderr)
        sys.exit(1)

    labels = load_labels(args.labels)

    print("=" * 80)
    print(" ExecuTorch Offline Waste Categorisation Engine")
    print(" Platform: Raspberry Pi (Offline Edge)")
    print("=" * 80)

    start_ram = get_ram_mb()
    print(f"[*] Base Process RAM: {start_ram:.2f} MB")
    print(f"[*] Loading ExecuTorch model: {args.model}...")

    t0 = time.perf_counter()
    module = exec_lib._load_for_executorch(args.model)
    load_time = (time.perf_counter() - t0) * 1000

    loaded_ram = get_ram_mb()
    model_size = os.path.getsize(args.model) / (1024 * 1024)
    print(f"[+] Model Size on Disk: {model_size:.2f} MB")
    print(f"[+] RAM after Load: {loaded_ram:.2f} MB (Allocated: {loaded_ram - start_ram:.2f} MB)")
    print(f"[+] Load Time: {load_time:.2f} ms")

    # Collect images
    image_items = collect_images(data_dir, labels, test_split_file=args.test_split)
    if args.limit:
        image_items = image_items[:args.limit]

    total_images = len(image_items)
    if total_images == 0:
        print(f"[!] No images found in '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    if args.test_split:
        print(f"[*] Evaluating strictly on HELD-OUT TEST SPLIT (15%): {total_images} images")
    else:
        print(f"[*] Evaluating across ALL {total_images} images from: {data_dir}")
    print("-" * 80)

    latencies = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    cat_correct = 0
    cat_total = 0

    has_ground_truth = any(gt != "unknown" for _, gt in image_items)

    start_eval_time = time.perf_counter()

    for idx, (img_path, gt_class) in enumerate(image_items, 1):
        tensor = preprocess(img_path)

        t_start = time.perf_counter()
        outputs = module.run_method("forward", (tensor,))
        lat_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(lat_ms)

        logits = np.array(outputs[0])[0]
        pred_idx = int(np.argmax(logits))
        pred_class = labels[pred_idx] if pred_idx < len(labels) else f"cls_{pred_idx}"

        if gt_class != "unknown":
            class_total[gt_class] += 1
            if pred_class == gt_class:
                class_correct[gt_class] += 1

            # Broad category evaluation
            pred_cat = "recyclable" if pred_class in RECYCLABLE_LABELS else "organic"
            gt_cat = "recyclable" if gt_class in RECYCLABLE_LABELS else "organic"
            cat_total += 1
            if pred_cat == gt_cat:
                cat_correct += 1

        # Periodic progress update
        if idx % 100 == 0 or idx == total_images:
            pct = (idx / total_images) * 100
            current_avg_lat = np.mean(latencies)
            if has_ground_truth and cat_total > 0:
                exact_acc = (sum(class_correct.values()) / cat_total) * 100
                print(
                    f" [{idx:>4}/{total_images}] {pct:>5.1f}% | "
                    f"Avg Latency: {current_avg_lat:>5.1f} ms | "
                    f"Exact Acc: {exact_acc:>5.1f}% | "
                    f"Bin Acc: {(cat_correct / cat_total * 100):>5.1f}%"
                )
            else:
                print(f" [{idx:>4}/{total_images}] {pct:>5.1f}% | Avg Latency: {current_avg_lat:>5.1f} ms")

    total_eval_wall_time = time.perf_counter() - start_eval_time
    peak_ram = get_ram_mb()

    print("-" * 80)
    print(" Performance & Throughput Metrics:")
    print(f"  - Total Images Evaluated : {total_images}")
    print(f"  - Total Evaluation Time  : {total_eval_wall_time:.1f} s")
    print(f"  - Average Inference Time : {np.mean(latencies):.2f} ms")
    print(f"  - Median Inference Time  : {np.median(latencies):.2f} ms")
    print(f"  - 95th Percentile Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"  - Min / Max Latency      : {np.min(latencies):.2f} ms / {np.max(latencies):.2f} ms")
    print(f"  - Real-time Throughput   : {1000 / np.mean(latencies):.1f} FPS (items/sec)")
    print(f"  - Peak RAM Footprint     : {peak_ram:.2f} MB")

    if has_ground_truth and cat_total > 0:
        total_exact = sum(class_correct.values())
        print("-" * 80)
        print(" Accuracy Metrics:")
        print(f"  - Overall Exact Accuracy : {total_exact}/{cat_total} ({(total_exact / cat_total * 100):.2f}%)")
        print(f"  - Societal Bin Accuracy  : {cat_correct}/{cat_total} ({(cat_correct / cat_total * 100):.2f}%) "
              f"(Recyclable vs Organic)")
        print("\n Per-Class Accuracy Breakdown:")
        print(f"  {'Class':<12} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
        print("  " + "-" * 44)
        for cls in sorted(labels):
            c_tot = class_total[cls]
            c_cor = class_correct[cls]
            c_acc = (c_cor / c_tot * 100) if c_tot > 0 else 0.0
            print(f"  {cls:<12} | {c_cor:<8} | {c_tot:<8} | {c_acc:>7.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()

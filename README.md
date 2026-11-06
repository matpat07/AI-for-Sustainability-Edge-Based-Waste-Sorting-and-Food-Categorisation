# ♻️ EcoSort Edge — Offline AI for Waste Classification

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-1.6%20(XNNPACK)-5B56DB)](https://pytorch.org/executorch)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-3B%2B%20Tested%20%26%20Verified-A22846?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![Accuracy](https://img.shields.io/badge/Held--out%20Accuracy-95.27%25-brightgreen)](audits/inference/v2.1.0_edge_modes_benchmark.md)
[![Hackathon](https://img.shields.io/badge/Namma%20AI%202026-Problem%20Statement%203-186A65)](#-hackathon-deliverables-ps3)

> **Namma AI Hackathon 2026 · PS3 — AI for Sustainability: Edge-Based Waste Sorting and Food Categorisation**

**EcoSort Edge** is a fully offline, high-accuracy, privacy-preserving Edge AI waste categorization system designed to run on low-power, constrained hardware (Raspberry Pi 3B+).

Trained and optimized with PyTorch and **ExecuTorch (XNNPACK backend)**, it achieves **95.27% 4-class classification accuracy** and **99.84% societal bin segregation accuracy** (Dry Recyclable vs Organic Wet Waste) while consuming only **~74 MB RAM** and **~3.4 W power**.

---

## 📂 Repository Layout

```
.
├── configs/                      # Data manifests and category mappings
│   ├── label_map.json            # 2-class baseline mapping
│   ├── label_map_2.json          # 4-class extended mapping
│   └── test_split.json           # 613 strictly held-out evaluation items
├── models/                       # Edge model binaries & labels
│   ├── waste_classifier.pte      # Compiled ExecuTorch XNNPACK model (8.48 MB)
│   ├── mobilenetv2_waste.pth     # PyTorch fine-tuned weights (9.16 MB)
│   └── labels.txt                # 4-class target label vocabulary
├── audits/                       # Versioned training & inference audit reports
│   ├── training/                 # Model training logs & metrics
│   │   ├── v2.0.0_mobilenetv2_4class.md
│   │   └── v2.0.0_mobilenetv2_4class.json
│   └── inference/                # On-device physical Raspberry Pi benchmarks
│       ├── v2.1.0_edge_modes_benchmark.md
│       └── v2.1.0_edge_modes_benchmark.json
├── notebooks/                    # Interactive development & training notebooks
│   └── waste_classifier_gpu.ipynb# Cloud GPU/NPU notebook with ExecuTorch export
├── src/                          # Modular core library
│   ├── dataset_audit.py          # Read-only dataset structure auditor
│   ├── evaluate.py               # Desktop CPU evaluation suite
│   └── model_loader.py           # Pretrained model registry & transforms
├── tests/                        # Offline unit test suite
│   └── test_pipeline.py
├── run_edge.sh                   # Edge runner with max, balanced, & eco profiles
├── eval.py                       # On-device edge evaluation runner
├── infer.py                      # Standalone single-image edge classifier
├── train.py                      # Fine-tuning & ExecuTorch compiler pipeline
├── setup_dataset.py              # Automated dataset downloader
├── requirements.txt              # Host Python dependencies
└── README.md                     # Project documentation
```

---

## ⚡ Edge Operational Modes & Benchmarks

Benchmarked directly on a physical **Raspberry Pi 3B+** (ARM Cortex-A53, 1.4 GHz, Debian GNU/Linux 13 trixie aarch64) across **613 strictly held-out test images**:

| Metric | Mode 1: MAX | Mode 2: BALANCED ⭐️ | Mode 3: ECO |
| :--- | :---: | :---: | :---: |
| **Command** | `./run_edge.sh eval --max` | `./run_edge.sh eval --balanced` | `./run_edge.sh eval --eco` |
| **CPU Allocation** | All 4 Cores (100% CPU) | 2 Cores (`taskset -c 0,1`) | 1 Core (`taskset -c 0`) |
| **Duty Cycle Delay** | `0 ms` (back-to-back) | `50 ms` sleep | `100 ms` sleep |
| **Inference Latency** | **78.15 ms** | **93.73 ms** | **162.94 ms** |
| **Throughput** | **12.8 FPS** | **10.7 FPS** | **6.1 FPS** |
| **Estimated Avg Power**| **~5.3 W** (1060 mA) | **~3.4 W** (680 mA) | **~2.6 W** (520 mA) |
| **Power Savings** | Baseline (0%) | **~36% Power Reduction** | **~51% Power Reduction** |
| **Energy / Inference** | **0.414 Joules** | **0.317 Joules** ⭐️ | **0.426 Joules** |
| **Peak RAM** | **73.63 MB** | **76.12 MB** | **73.58 MB** |
| **Thermal Profile** | +6.0°C rise in 6s (Hot) | Cool / Stable | Barely above idle |

> ⭐️ **BALANCED mode** is the production default: it only adds **~15.6 ms** latency while halving CPU utilization and saving **36% power**, keeping cores 2 & 3 free for camera I/O and networking.

---

## 🎯 Model Accuracy

| Waste Class | Correct | Total | Exact Accuracy | Societal Bin Category |
| :--- | :---: | :---: | :---: | :--- |
| **`biodegradable`** | 129 | 130 | **99.23%** | 🍂 Organic (Wet Waste) |
| **`metal`** | 169 | 173 | **97.69%** | ♻️ Recyclable (Dry Waste) |
| **`glass`** | 153 | 163 | **93.87%** | ♻️ Recyclable (Dry Waste) |
| **`plastic`** | 133 | 147 | **90.48%** | ♻️ Recyclable (Dry Waste) |
| **Total / Overall** | **584** | **613** | **95.27%** | **99.84% Bin Accuracy** |

---

## 🚀 Quick Start Guide

### 1. Setup Local Environment
```bash
git clone https://github.com/parth/nama.git
cd nama
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python3 -m unittest discover tests
```

### 3. Deploy & Run on Raspberry Pi
```bash
# SSH into Raspberry Pi
ssh parth@192.168.1.11
cd ~/waste_edge_ai

# Single-Image Inference
./run_edge.sh infer "dataset/waste_4class/plastic/pet (105).jpg"

# Run Balanced Batch Evaluation (50ms pacing, 2 cores)
./run_edge.sh eval

# Run Eco Evaluation (100ms pacing, 1 core)
./run_edge.sh eval --eco
```

---

## 🏆 Hackathon Deliverables (PS3)

- [x] **Lightweight Edge Classifier:** MobileNetV2 quantized/lowered to ExecuTorch XNNPACK (`8.48 MB`).
- [x] **Zero Cloud Dependency:** 100% offline inference on Raspberry Pi with local C++ shims.
- [x] **Verified Accuracy:** **95.27%** exact accuracy, **99.84%** societal segregation accuracy.
- [x] **Audited & Reproducible:** Complete audit trails versioned in [`audits/`](audits/).

# ⚡ Inference Audits Registry

This directory contains versioned edge deployment logs, latency profiles, memory footprints, and classification accuracy benchmarks measured directly on physical target edge hardware (Raspberry Pi).

## Version Index

| Version | Target Device | Model | Classes | Accuracy | Latency Range | FPS Range | Peak RAM | Benchmark Report |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`v2.1.0`** | Raspberry Pi 3B+/4 (ARM64) | `waste_classifier.pte` | 4 | **95.27%** | **78.2 - 162.9 ms** | **6.1 - 12.8 FPS** | **76.1 MB** | [Multi-Mode Report (MD)](v2.1.0_edge_modes_benchmark.md) · [JSON](v2.1.0_edge_modes_benchmark.json) |
| **`v2.0.0`** | Raspberry Pi 3B+/4 (ARM64) | `waste_classifier.pte` | 4 | **95.27%** | **75.6 ms** | **13.2 FPS** | **73.6 MB** | [Report (MD)](v2.0.0_edge_pi_benchmark.md) · [JSON](v2.0.0_edge_pi_benchmark.json) |
| **`v1.0.0`** | Raspberry Pi 3B+/4 (ARM64) | `waste_classifier_6class.pte` | 6 | **89.21%** | **76.3 ms** | **13.1 FPS** | **56.8 MB** | 6-Class TrashNet Baseline |

# 📊 Training Audits Registry

This directory contains versioned audit reports, parameter logs, training metrics, and cryptographic checksums for all trained waste classification models.

## Version Index

| Version | Date | Model | Target Classes | Train Acc | Val Acc | Target Platform | Audit Report |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **`v2.0.0`** | 2026-11-06 | `mobilenetv2_waste_4class` | 4 (`biodegradable`, `glass`, `metal`, `plastic`) | **96.63%** | **95.27%** | ExecuTorch XNNPACK (Raspberry Pi / Edge) | [Report (MD)](v2.0.0_mobilenetv2_4class.md) · [JSON](v2.0.0_mobilenetv2_4class.json) |
| **`v1.0.0`** | 2026-11-05 | `mobilenetv2_waste_6class` | 6 (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) | **75.10%** | **89.21%** | ExecuTorch XNNPACK (Raspberry Pi / Edge) | Initial Baseline (TrashNet) |

---

## Audit File Schema

Each training audit contains:
- **`*.json`**: Machine-readable specification with exact cryptographic SHA-256 hashes of all artifacts (`.pte`, `.pth`, `labels.txt`, `test_split.json`), hardware details, hyperparameter specifications, and epoch loss/accuracy progression.
- **`*.md`**: Formatted human-readable audit report summarizing dataset splits, class distributions, architecture decisions, and edge compilation notes.

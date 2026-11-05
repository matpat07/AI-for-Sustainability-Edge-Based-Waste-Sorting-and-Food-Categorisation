# AI-for-Sustainability-Edge-Based-Waste-Sorting-and-Food-Categorisation

Phase 1: read-only local dataset auditing and pretrained PyTorch CPU evaluation.
No training, Raspberry Pi deployment or ExecuTorch conversion is included.
Do not rerun `setup_dataset.py`: the dataset already exists locally. That script
is unchanged. No dataset files are tracked by Git at implementation time.

## Installation (Windows PowerShell, repository root)

Use an accessible Python 3.11+ installation and the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.dataset_audit --data-dir '.\dataset'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If the existing environment cannot launch, install a working Python interpreter
and create a separate environment without overwriting it:

```powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r requirements.txt
```

Use that environment's Python for the commands below. The execution environment
used for implementation could not launch either the existing venv or the Windows
Store Python executable, so image validity, tests and real inference remain unverified.

## Inspected local data

`dataset/Waste Images/<class>/<image>` contains 8,346 candidate files, no CSVs:

| Exact class | Candidate files |
| --- | ---: |
| Aluminium | 1019 |
| Carton | 533 |
| E-waste | 1029 |
| Glass | 1089 |
| Organic_Waste | 1053 |
| Paper_and_Cardboard | 1194 |
| Plastics | 1035 |
| Textiles | 830 |
| Wood | 564 |

Extensions: 8,139 `.jpg`, 48 `.jpeg`, 48 `.png`, 111 `.jfif`. These are filesystem
counts, not verified decoded-image counts. The audit reports actual decoded formats,
valid class counts and unreadable paths without changing any images.

For an external dataset use `--data-dir 'D:\Local Data\Waste Images'`.
Class-folder discovery accepts a flat class-parent layout with unambiguous wrapper
folders. Flat image directories are unlabelled and produce predictions without metrics.
Nested train/test layouts require selecting the split's class-parent directory.
CSV annotations require explicit `--annotations 'D:\Local Data\labels.csv'`, with
`path,label` columns (override with `--path-column` and `--label-column`). Paths are
absolute inside `--data-dir` or relative to it; duplicate paths and escapes are rejected.
CSV labels take precedence over folders. CSVs found during folder discovery require
explicit selection, avoiding accidental annotation guessing.

## Verified model adapters

Both Hugging Face model cards declare MIT for the selected waste classifiers:

| Hugging Face ID | Checkpoint | Index order |
| --- | --- | --- |
| [tonghahaha/trashnet-resnet18](https://huggingface.co/tonghahaha/trashnet-resnet18) | `best_resnet18_trashnet.pth` | cardboard, glass, metal, paper, plastic, trash |
| [TheHelltaker/smart-waste-segregator](https://huggingface.co/TheHelltaker/smart-waste-segregator) | `mobilenetv3-small/best_model.pth` | metal, plastic, glass, biodegradable |

ResNet18 uses a 64-unit custom head, Pillow bilinear 224x224 resize, RGB tensor
scaled to [0,1], **no ImageNet normalization**, following the author's
[inference code](https://github.com/yutongyu-ai/Trash_Classification_System/blob/main/backend/inference.py)
and HF architecture example. The runtime calls index 5 `General Waste`; this
adapter retains the HF card's canonical `trash` label for that index.

MobileNetV3-Small uses a `model.` wrapper and replaces the final classifier layer
with four outputs, following the author's
[export code](https://github.com/divyanshuklai/smart-waste-segregator/blob/main/rpi-mobilenet/export.py).
Preprocessing follows the author's [inference code](https://github.com/divyanshuklai/smart-waste-segregator/blob/main/rpi-mobilenet/inference.py):
OpenCV RGB conversion, linear 224x224 resize, /255, ImageNet mean/std normalization.
The HF card explicitly corrects conflicting class lists in older runtime scripts.
YOLO files in that repository are stock models, not waste-trained; they are unsupported
here and have a separate AGPL-3.0 license. Checkpoint formats that fail strict loading
are rejected; unsafe pickle fallback and partial loading are never used.

## Compatibility and evaluation

Neither model represents all nine local classes. This command intentionally fails
before downloading weights, listing incompatible categories:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --data-dir '.\dataset' --model tonghahaha/trashnet-resnet18
```

No spelling changes, merges or semantic mappings are automatic. For a conservative
comparison on the existing `Glass` category, explicitly approve the exact-name mapping
by creating this local JSON file, then use the same selection for both models:

```powershell
New-Item -ItemType Directory -Force results/local | Out-Null
'{"Glass":"glass"}' | Set-Content -Encoding utf8 results/local/glass-map.json
.\.venv\Scripts\python.exe -m src.evaluate --data-dir '.\dataset' --model tonghahaha/trashnet-resnet18 --classes Glass --label-map results/local/glass-map.json
.\.venv\Scripts\python.exe -m src.evaluate --data-dir '.\dataset' --model TheHelltaker/smart-waste-segregator --classes Glass --label-map results/local/glass-map.json --compare-to 'results/local/<first-run>/summary.json'
```

Replace `<first-run>` with the printed ResNet run folder. A one-class evaluation
measures only Glass performance, not general waste sorting accuracy. Broader evaluations
require independently justified one-to-one labels; `Paper_and_Cardboard` cannot map
to both paper and cardboard, and the missing categories cannot be guessed or merged.
All model outputs remain active: an output outside the selected categories is an error.
`--compare-to` requires identical paths, image SHA256s, original/target labels and target
class sets. Comparing models with different native output spaces is therefore a
comparison on the stated target subset, not overall classifier performance.

First inference downloads only the selected model checkpoint into ignored `.hf_cache/`.
Use `--revision <HF-commit-SHA>` to pin a checkpoint, and `--offline` after caching.
No dataset download occurs. All selected images must pass decoding; corrupt files
abort evaluation rather than silently changing the comparison set.

Each unique `results/local/<run>/` contains per-image `predictions.csv`,
`summary.json` with accuracy, per-class precision/recall/F1, macro and weighted metrics,
and labelled confusion matrix CSV/PNG. Unlabelled data has no metrics or matrix.
The summary records checkpoint SHA256, image manifest/hash, label mapping, package
versions and CPU settings. Warm-up defaults to five iterations, followed by one
forward pass per image, batch size one and one CPU thread. Forward-pass latency excludes
decoding/preprocessing, hashing, warm-up, checkpoint loading and softmax/postprocessing;
preprocessing latency is reported separately. No timing is a Raspberry Pi claim.

Datasets, common image extensions, weights, caches, environments and results are ignored
by Git. Keep external data local; do not force-add ignored files. Before sharing code,
inspect `git status --short` and `git ls-files dataset data datasets results weights checkpoints`.

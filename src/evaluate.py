"""Reproducible local CPU evaluation. Run with python -m src.evaluate."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .dataset_audit import add_data_arguments, audit, discover
from .model_loader import get_spec, load_model


def select_samples(samples, classes, mapping, model_classes):
    available = {s.label for s in samples if s.label is not None}
    if classes and not set(classes).issubset(available):
        raise ValueError(f'Unknown selected classes: {set(classes) - available}')
    selected = [s for s in samples if classes is None or s.label in classes]
    if not selected:
        raise ValueError('No evaluation examples selected')
    labels = {s.label for s in selected if s.label is not None}
    unknown_keys = set(mapping) - available
    if unknown_keys:
        raise ValueError(f'Label map contains unknown dataset classes: {unknown_keys}')
    targets = {label: mapping.get(label, label) for label in labels}
    if len(set(targets.values())) != len(targets):
        raise ValueError('Many-to-one class merging is not supported in Phase 1')
    incompatible = {label: target for label, target in targets.items() if target not in model_classes}
    if incompatible:
        raise ValueError(f'Incompatible categories: {incompatible}. Model classes: {model_classes}. '
                         'Use --classes and an explicit --label-map JSON only for equivalent categories; no names are guessed.')
    return selected, targets


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', default='main', help='Prefer an immutable Hugging Face commit SHA')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--classes', nargs='+', help='Explicit dataset class subset; all images in each selected class')
    parser.add_argument('--label-map', type=Path, help='JSON object of exact dataset labels to exact model labels')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--compare-to', type=Path, help='Previous summary.json: require same examples and target classes')
    args = parser.parse_args()
    if args.warmup < 1 or args.threads < 1:
        parser.error('--warmup and --threads must be positive')
    try:
        spec = get_spec(args.model)
        root, samples = discover(args.data_dir, args.annotations, args.path_column, args.label_column)
        mapping = json.loads(args.label_map.read_text(encoding='utf-8-sig')) if args.label_map else {}
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError('--label-map must contain a JSON object of string labels')
        selected, targets = select_samples(samples, args.classes, mapping, spec.classes)
        valid, audit_report = audit(selected)
        if audit_report['unreadable']:
            raise ValueError(f"Unreadable selected images ({len(audit_report['unreadable'])}); run src.dataset_audit for paths. No examples silently skipped.")
        manifest = [{'path': s.path.relative_to(root).as_posix(), 'sha256': sha256(s.path),
                     'dataset_label': s.label, 'target_label': targets.get(s.label)} for s in valid]
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        target_classes = sorted(set(targets.values()))
        if args.compare_to:
            previous = json.loads(args.compare_to.read_text(encoding='utf-8'))
            if previous['dataset_sha256'] != fingerprint or previous['target_classes'] != target_classes:
                raise ValueError('Comparison rejected: examples, contents, ground truth or target classes differ')
        import torch
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        model, preprocess, checkpoint = load_model(args.model, args.revision, args.offline)
        rows = []
        with torch.inference_mode():
            warmup_input = preprocess(valid[0].path)
            for _ in range(args.warmup):
                model(warmup_input)
            for sample in valid:
                start = perf_counter()
                tensor = preprocess(sample.path)
                preprocessing_ms = (perf_counter() - start) * 1000
                start = perf_counter()
                logits = model(tensor)
                inference_ms = (perf_counter() - start) * 1000
                if tuple(logits.shape) != (1, len(spec.classes)) or not torch.isfinite(logits).all():
                    raise ValueError('Checkpoint returned an invalid shape or nonfinite logits')
                probabilities = logits.softmax(dim=1)[0]
                index = int(probabilities.argmax())
                rows.append({'path': sample.path.relative_to(root).as_posix(), 'dataset_label': sample.label,
                             'target_label': targets.get(sample.label), 'predicted_label': spec.classes[index],
                             'predicted_index': index, 'confidence': float(probabilities[index]),
                             'preprocessing_ms': preprocessing_ms, 'inference_ms': inference_ms})
        summary = {'model': args.model, 'requested_revision': args.revision, 'checkpoint_path': str(checkpoint),
                   'checkpoint_sha256': sha256(checkpoint), 'model_spec': asdict(spec),
                   'dataset_root': str(root), 'dataset_sha256': fingerprint, 'manifest': manifest,
                   'target_classes': target_classes, 'label_map': mapping, 'audit': audit_report,
                   'device': 'cpu', 'batch_size': 1, 'threads': args.threads, 'warmup_iterations': args.warmup,
                   'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': {name: importlib.metadata.version(name) for name in
                                ('torch', 'torchvision', 'Pillow', 'huggingface-hub', 'numpy', 'scikit-learn', 'opencv-python-headless', 'matplotlib')},
                   'average_inference_ms': sum(r['inference_ms'] for r in rows) / len(rows),
                   'average_preprocessing_ms': sum(r['preprocessing_ms'] for r in rows) / len(rows)}
        output = Path('results/local') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + args.model.replace('/', '__') + '_' + uuid4().hex[:8])
        output.mkdir(parents=True)
        with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        labelled = [row for row in rows if row['target_label'] is not None]
        if labelled:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            truth = [r['target_label'] for r in labelled]
            predictions = [r['predicted_label'] for r in labelled]
            # Retain all model outputs; predictions outside selected targets count as errors.
            report = classification_report(truth, predictions, labels=list(spec.classes), output_dict=True, zero_division=0)
            target_report = {name: report[name] for name in target_classes}
            summary['metrics'] = {'accuracy': accuracy_score(truth, predictions), 'per_class': target_report,
                                  'macro_precision': sum(report[c]['precision'] for c in target_classes) / len(target_classes),
                                  'macro_recall': sum(report[c]['recall'] for c in target_classes) / len(target_classes),
                                  'macro_f1': sum(report[c]['f1-score'] for c in target_classes) / len(target_classes),
                                  'weighted_average': report['weighted avg']}
            matrix = confusion_matrix(truth, predictions, labels=list(spec.classes))
            with (output / 'confusion_matrix.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['true/predicted', *spec.classes])
                writer.writerows([label, *row] for label, row in zip(spec.classes, matrix.tolist()))
            display = ConfusionMatrixDisplay(matrix, display_labels=spec.classes)
            display.plot(xticks_rotation=45)
            plt.tight_layout()
            plt.savefig(output / 'confusion_matrix.png', dpi=150)
            plt.close()
        else:
            summary['metrics'] = None
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Results: {output}\nMean forward pass: {summary["average_inference_ms"]:.3f} ms; preprocessing: {summary["average_preprocessing_ms"]:.3f} ms')
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

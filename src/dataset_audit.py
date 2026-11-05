"""Read-only discovery and decoding audit; never changes source images."""
import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.jfif', '.png', '.webp', '.bmp', '.tif', '.tiff'}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str | None


def discover(data_dir, annotations=None, path_column='path', label_column='label'):
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    if annotations:
        samples = []
        with Path(annotations).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {path_column, label_column}.issubset(reader.fieldnames or []):
                raise ValueError(f'CSV needs columns {path_column!r}, {label_column!r}')
            for row in reader:
                if not row[path_column].strip() or not row[label_column].strip():
                    raise ValueError('CSV contains an empty path or label')
                path = (root / row[path_column]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f'CSV image must be inside --data-dir: {path}')
                samples.append(Sample(path, row[label_column].strip()))
        if len({s.path for s in samples}) != len(samples):
            raise ValueError('CSV contains duplicate image paths')
        return root, sorted(samples, key=lambda s: s.path.as_posix())
    csvs = list(root.rglob('*.csv'))
    if csvs:
        raise ValueError(f'Annotation CSV found: {csvs[0]}. Select it with --annotations; layout will not be guessed.')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('No supported image files found')
    # Peel unambiguous wrapper directories (the inspected dataset has "Waste Images").
    while all(len(p.relative_to(root).parts) >= 3 for p in paths) and all(p.relative_to(root).parts[0] == paths[0].relative_to(root).parts[0] for p in paths):
        root = root / paths[0].relative_to(root).parts[0]
    depths = {len(p.relative_to(root).parts) for p in paths}
    if depths == {1}:
        return root, [Sample(p, None) for p in paths]
    if depths != {2}:
        raise ValueError('Ambiguous nested layout. Set --data-dir to the class-folder parent or provide --annotations.')
    return root, [Sample(p, p.parent.name) for p in paths]


def audit(samples):
    valid, unreadable = [], []
    formats = Counter()
    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                image.verify()
            with Image.open(sample.path) as image:
                image.convert('RGB').load()
                formats[image.format] += 1
            valid.append(sample)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            unreadable.append({'path': str(sample.path), 'error': str(exc)})
    return valid, {'candidate_images': len(samples), 'valid_images': len(valid),
                   'class_distribution': dict(sorted(Counter(s.label for s in valid if s.label is not None).items())),
                   'unlabelled_images': sum(s.label is None for s in valid),
                   'decoded_formats': dict(formats), 'unreadable': unreadable}


def add_data_arguments(parser):
    parser.add_argument('--data-dir', default='dataset')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--path-column', default='path')
    parser.add_argument('--label-column', default='label')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    args = parser.parse_args()
    try:
        root, samples = discover(args.data_dir, args.annotations, args.path_column, args.label_column)
        _, report = audit(samples)
        print(json.dumps({'resolved_root': str(root), **report}, indent=2))
    except ValueError as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

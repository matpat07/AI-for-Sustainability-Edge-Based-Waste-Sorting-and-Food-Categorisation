"""Offline checks; temporary synthetic images never touch the local dataset."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset_audit import Sample, audit, discover
from src.evaluate import select_samples
from src.model_loader import get_spec


class PipelineTests(unittest.TestCase):
    def test_wrapper_formats_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'Waste Images' / 'Glass'
            folder.mkdir(parents=True)
            # Add a second class so peeling wrappers is unambiguous.
            other = folder.parent / 'Wood'
            other.mkdir()
            Image.new('RGB', (4, 4)).save(folder / 'good.jfif', format='JPEG')
            (other / 'bad.png').write_bytes(b'broken')
            root, samples = discover(directory)
            valid, report = audit(samples)
            self.assertEqual(root.name, 'Waste Images')
            self.assertEqual(len(valid), 1)
            self.assertEqual(report['class_distribution'], {'Glass': 1})
            self.assertEqual(len(report['unreadable']), 1)

    def test_csv_escape_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / 'labels.csv'
            with annotation.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerows([['path', 'label'], ['../outside.jpg', 'glass']])
            with self.assertRaisesRegex(ValueError, 'inside'):
                discover(directory, annotation)
            annotation.write_text('path,label\nx.jpg,glass\nx.jpg,glass\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                discover(directory, annotation)

    def test_single_class_preserves_label(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'Glass'
            folder.mkdir()
            Image.new('RGB', (4, 4)).save(folder / 'image.png')
            _, samples = discover(directory)
            self.assertEqual(samples[0].label, 'Glass')

    def test_compatibility_never_guesses(self):
        samples = [Sample(Path('Glass/a.jpg'), 'Glass'), Sample(Path('Wood/a.jpg'), 'Wood')]
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            select_samples(samples, None, {}, ('glass',))
        chosen, targets = select_samples(samples, ['Glass'], {'Glass': 'glass'}, ('glass',))
        self.assertEqual(len(chosen), 1)
        self.assertEqual(targets, {'Glass': 'glass'})
        with self.assertRaisesRegex(ValueError, 'merging'):
            select_samples(samples, None, {'Glass': 'glass', 'Wood': 'glass'}, ('glass',))

    def test_unsupported_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            get_spec('unknown/model')


if __name__ == '__main__':
    unittest.main()

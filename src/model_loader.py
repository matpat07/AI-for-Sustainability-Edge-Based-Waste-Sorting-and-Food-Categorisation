"""Author-documented adapters. Strictly load trained weights before returning."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    classes: tuple[str, ...]
    architecture: str
    preprocessing: str
    license: str = 'MIT'


MODELS = {
    'tonghahaha/trashnet-resnet18': ModelSpec(
        'best_resnet18_trashnet.pth', ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'),
        'resnet18 with Linear(512,64), ReLU, Dropout(0.5), Linear(64,6)',
        'RGB; Pillow bilinear resize 224x224; float32 /255; no normalization'),
    'TheHelltaker/smart-waste-segregator': ModelSpec(
        'mobilenetv3-small/best_model.pth', ('metal', 'plastic', 'glass', 'biodegradable'),
        'WasteClassifier.model = mobilenet_v3_small; final Linear(1024,4)',
        'OpenCV BGR decode to RGB; INTER_LINEAR 224x224; /255; ImageNet mean/std'),
}


def get_spec(model_id):
    if model_id not in MODELS:
        raise ValueError(f'Unsupported checkpoint {model_id!r}. Supported: {list(MODELS)}')
    return MODELS[model_id]


def load_model(model_id, revision='main', offline=False):
    import torch
    from torch import nn
    from torchvision import models, transforms
    from huggingface_hub import hf_hub_download

    spec = get_spec(model_id)
    if model_id == 'tonghahaha/trashnet-resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 6))
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        def preprocess(path):
            from PIL import Image
            with Image.open(path) as image:
                return transform(image.convert('RGB')).unsqueeze(0)
    else:
        class WasteClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = models.mobilenet_v3_small(weights=None)
                self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 4)

            def forward(self, value):
                return self.model(value)

        model = WasteClassifier()

        def preprocess(path):
            import cv2
            import numpy as np
            # imdecode also supports Windows paths with non-ASCII characters.
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'OpenCV cannot decode {path}')
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
            return torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)

    checkpoint = Path(hf_hub_download(repo_id=model_id, filename=spec.filename, revision=revision,
                                    cache_dir='.hf_cache', local_files_only=offline))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f'Unsupported checkpoint structure for {model_id}; no partial/random weights used: {exc}') from exc
    model.cpu().eval()
    return model, preprocess, checkpoint

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.stats import entropy


_INCEPTION_MODEL = None


def get_inception_model(device="cpu"):
    global _INCEPTION_MODEL
    if _INCEPTION_MODEL is None:
        weights = Inception_V3_Weights.DEFAULT
        _INCEPTION_MODEL = inception_v3(weights=weights, transform_input=False).to(device)
        _INCEPTION_MODEL.eval()
    else:
        _INCEPTION_MODEL = _INCEPTION_MODEL.to(device)
    return _INCEPTION_MODEL


def inception_score(
    imgs,
    batch_size=32,
    splits=10,
    device="cpu"
):
    assert isinstance(imgs, torch.Tensor)
    assert imgs.dim() == 4
    assert imgs.size(1) == 3

    N = imgs.size(0)
    assert N > 0

    inception = get_inception_model(device)
    upsample = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False).to(device)

    # Images are expected in [0,1] range for Inception v3
    imgs = imgs.to(device=device, dtype=torch.float32)
    imgs = torch.clamp(imgs, 0.0, 1.0)

    # ImageNet normalization for pretrained Inception v3
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    preds = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = imgs[i:i + batch_size]
            batch = upsample(batch)
            batch = (batch - mean) / std
            outputs = inception(batch)
            probs = F.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    split_scores = []
    effective_splits = min(splits, N)

    for k in range(effective_splits):
        start = k * (N // effective_splits)
        end = (k + 1) * (N // effective_splits) if k < effective_splits - 1 else N
        part = preds[start:end]
        if len(part) == 0:
            continue

        py = np.mean(part, axis=0)

        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            scores.append(entropy(pyx, py))

        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))
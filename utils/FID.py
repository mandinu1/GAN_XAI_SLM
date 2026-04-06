import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)

        # remove final classification layer
        model.fc = nn.Identity()

        self.model = model.to(device)
        self.model.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = x.to(device=device, dtype=torch.float32)
        x = torch.clamp(x, 0.0, 1.0)

        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = (x - mean) / std

        return self.model(x)


_EXTRACTOR = None


def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = InceptionFeatureExtractor()
    return _EXTRACTOR


def get_features(images, model, batch_size=32):
    features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            feat = model(batch)

        features.append(feat.cpu())

    features = torch.cat(features, dim=0)
    return features.numpy()


def calculate_fid(real_images, fake_images, batch_size=32):
    extractor = get_extractor()

    real_images = real_images.to(dtype=torch.float32)
    fake_images = fake_images.to(dtype=torch.float32)

    # Resize images for Inception v3
    real_images = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)
    fake_images = F.interpolate(fake_images, size=(299, 299), mode="bilinear", align_corners=False)

    # Extract features in batches
    real_features = get_features(real_images, extractor, batch_size=batch_size)
    fake_features = get_features(fake_images, extractor, batch_size=batch_size)

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake

    # Numerical stability
    cov_prod = sigma_real @ sigma_fake
    covmean, _ = linalg.sqrtm(cov_prod + 1e-6 * np.eye(sigma_real.shape[0]), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)
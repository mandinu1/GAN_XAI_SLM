import torch
import torch.nn.functional as F
import cv2
import numpy as np


class SaliencyMap:
    def __init__(self, model):
        self.model = model

    def generate(self, images):
        images = images.clone().detach().requires_grad_(True)

        self.model.zero_grad()

        output = self.model(images)
        loss = output.mean()
        loss.backward()

        saliency = images.grad.abs()
        saliency = saliency.mean(dim=1)  # smoother than max

        saliency_min = saliency.view(saliency.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency.detach()


def overlay_saliency(image, saliency):
    
    image = image.detach().cpu()
    image = image.permute(1, 2, 0).numpy()
    image = (image + 1) / 2

    saliency = saliency.detach().cpu().numpy()
    saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = 0.5 * heatmap + 0.5 * image
    overlay = np.clip(overlay, 0, 1)

    return overlay
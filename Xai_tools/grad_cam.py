import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAM:
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, images):
        images = images.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        output = self.model(images)
        loss = output.mean()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam


def overlay_cam(image, cam):
    

    # Convert image to numpy
    image = image.detach().cpu()
    image = image.permute(1, 2, 0).numpy()
    image = (image + 1) / 2  # denormalize to [0,1]

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = 0.5 * heatmap + 0.5 * image
    overlay = np.clip(overlay, 0, 1)

    return overlay


def extract_regions_from_map(cam, threshold=0.6, min_area=20, top_k=5):
    """
    Extract important regions from a normalized heatmap.

    Args:
        cam: torch.Tensor [H, W] or numpy array [H, W] in range [0, 1]
        threshold: float threshold for selecting salient regions
        min_area: minimum contour area to keep
        top_k: maximum number of regions to return

    Returns:
        list of dicts with bbox, centroid, area, and score
    """
    if isinstance(cam, torch.Tensor):
        cam_np = cam.detach().cpu().numpy()
    else:
        cam_np = cam

    cam_np = np.asarray(cam_np, dtype=np.float32)
    cam_np = np.clip(cam_np, 0.0, 1.0)

    binary = (cam_np >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = cam_np[y:y + h, x:x + w]
        score = float(roi.mean()) if roi.size > 0 else 0.0

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)

        regions.append({
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "centroid": {
                "x": round(cx, 2),
                "y": round(cy, 2)
            },
            "area": round(area, 2),
            "score": round(score, 4)
        })

    regions = sorted(regions, key=lambda r: r["score"], reverse=True)
    return regions[:top_k]
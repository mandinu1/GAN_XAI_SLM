import numpy as np
import torch
import shap
from skimage.measure import label, regionprops


class DiscriminatorWrapper(torch.nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, x):
        out = self.discriminator(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out


def extract_shap_regions(shap_map, min_area=20, top_k=5, threshold_ratio=0.6):
    """
    Convert a SHAP attribution map into structured important regions.
    """
    if shap_map.ndim == 3:
        shap_map = np.mean(np.abs(shap_map), axis=2)
    else:
        shap_map = np.abs(shap_map)

    if shap_map.max() <= 0:
        return []

    threshold = float(shap_map.max()) * float(threshold_ratio)
    binary = (shap_map >= threshold).astype(np.uint8)
    labeled = label(binary)

    regions = []
    for region in regionprops(labeled, intensity_image=shap_map):
        if region.area < min_area:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        cy, cx = region.centroid
        score = float(region.mean_intensity)

        regions.append({
            "bbox": {
                "x": int(min_col),
                "y": int(min_row),
                "width": int(max_col - min_col),
                "height": int(max_row - min_row)
            },
            "centroid": {
                "x": round(float(cx), 2),
                "y": round(float(cy), 2)
            },
            "area": int(region.area),
            "score": round(score, 4)
        })

    regions = sorted(regions, key=lambda r: r["score"], reverse=True)
    return regions[:top_k]


def run_shap(dcgan_model, background, test_image, device=None, min_area=20, top_k=5, threshold_ratio=0.6):
    """
    background: numpy array [N, H, W, C] in [0,1]
    test_image: numpy array [1, H, W, C] in [0,1]

    Returns:
        dict with:
        - shap_values: raw SHAP values
        - shap_map: condensed attribution map for visualization/regions
        - important_regions: structured regions
    """
    if device is None:
        device = next(dcgan_model.D.parameters()).device

    background_t = torch.from_numpy(background).permute(0, 3, 1, 2).float().to(device)
    test_image_t = torch.from_numpy(test_image).permute(0, 3, 1, 2).float().to(device)

    # match discriminator training range
    background_t = background_t * 2 - 1
    test_image_t = test_image_t * 2 - 1

    wrapped_discriminator = DiscriminatorWrapper(dcgan_model.D).to(device)
    wrapped_discriminator.eval()

    explainer = shap.DeepExplainer(wrapped_discriminator, background_t)
    shap_values = explainer.shap_values(test_image_t)

    raw = shap_values[0] if isinstance(shap_values, list) else shap_values
    raw = np.array(raw)

    # Normalize to [H, W, C] where possible
    if raw.ndim == 4:
        # [1, C, H, W] -> [H, W, C]
        shap_map = np.transpose(raw[0], (1, 2, 0))
    elif raw.ndim == 3:
        # [C, H, W] -> [H, W, C]
        shap_map = np.transpose(raw, (1, 2, 0))
    elif raw.ndim == 2:
        shap_map = raw
    else:
        shap_map = np.squeeze(raw)

    important_regions = extract_shap_regions(
        shap_map,
        min_area=min_area,
        top_k=top_k,
        threshold_ratio=threshold_ratio,
    )

    return {
        "shap_values": shap_values,
        "shap_map": shap_map,
        "important_regions": important_regions,
    }
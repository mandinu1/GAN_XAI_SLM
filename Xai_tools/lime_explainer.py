import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
import torch


def extract_lime_regions(mask, weights_dict, min_area=20, top_k=5):
    
    binary = (mask > 0).astype(np.uint8)
    labeled = label(binary)

    regions = []
    for region in regionprops(labeled):
        if region.area < min_area:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        cy, cx = region.centroid

        region_mask = labeled == region.label
        segment_ids = np.unique(mask[region_mask])
        segment_ids = [int(s) for s in segment_ids if int(s) != 0]

        if len(segment_ids) > 0:
            score = float(np.mean([weights_dict.get(seg_id, 0.0) for seg_id in segment_ids]))
        else:
            score = 0.0

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

    regions = sorted(regions, key=lambda r: abs(r["score"]), reverse=True)
    return regions[:top_k]


def run_lime(dcgan_model, image, device=None, num_samples=500, min_area=20, top_k=5):
   
    if device is None:
        device = next(dcgan_model.D.parameters()).device

    def discriminator_predict(images):
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)

        # Convert [0,1] -> [-1,1] to match training range
        images = images * 2 - 1

        with torch.no_grad():
            preds = dcgan_model.D(images)

        preds = preds.detach().cpu().numpy().reshape(-1, 1)
        return np.hstack([1 - preds, preds])

    image_np = (image.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    image_np = np.clip(image_np, 0, 1).astype(np.float32)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image=image_np,
        classifier_fn=discriminator_predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples
    )

    label_id = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label_id,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    local_weights = dict(explanation.local_exp[label_id])
    important_regions = extract_lime_regions(mask, local_weights, min_area=min_area, top_k=top_k)
    visualization = mark_boundaries(temp, mask)

    return {
        "image": visualization,
        "mask": mask,
        "important_regions": important_regions,
        "raw_weights": {str(k): float(v) for k, v in local_weights.items()}
    }
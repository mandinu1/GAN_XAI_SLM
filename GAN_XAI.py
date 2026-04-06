import os
import json
import argparse
import re
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import utils
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.dcgan import DCGAN_MODEL
from utils.data_loader import get_data_loader
from Xai_tools.grad_cam import GradCAM, overlay_cam, extract_regions_from_map
from Xai_tools.saliency_map import SaliencyMap, overlay_saliency
from Xai_tools.lime_explainer import run_lime
from Xai_tools.shap_explainer import run_shap


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate XAI outputs for GAN discriminator decisions."
    )

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "places365", "stl10"])
    parser.add_argument("--dataroot", type=str, required=True, help="Path containing train/ and val/ (or test/)")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--run_name", type=str, required=True, help="Experiment run name")
    parser.add_argument("--load_G", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--load_D", type=str, required=True, help="Path to discriminator checkpoint")

    parser.add_argument("--target_class", type=str, default=None, help="Use one class only, e.g. mountain")
    parser.add_argument("--source", type=str, default="generated", choices=["generated", "real"])
    parser.add_argument("--num_images", type=int, default=1, help="How many images to process")

    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for important regions")
    parser.add_argument("--min_area", type=int, default=20, help="Minimum region area")
    parser.add_argument("--image_size", type=int, default=256, help="Visualization size")
    parser.add_argument("--show_images", action="store_true", help="Display input image and Grad-CAM overlay")
    parser.add_argument("--generate_slm_explanations", action="store_true", help="Generate an SLM explanation for each processed image")
    parser.add_argument("--slm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="SLM model name")
    parser.add_argument("--slm_max_new_tokens", type=int, default=600, help="Max new tokens for SLM explanation")
    parser.add_argument("--force_slm_cpu", action="store_true", help="Force SLM generation on CPU")

    return parser.parse_args()


def display_gradcam_result(input_image_path, gradcam_overlay_path, summary_text, image_id,
                           saliency_overlay_path=None, lime_path=None, shap_overlay_path=None,
                           save_path=None, show_plot=True):
    image_paths = [
        (input_image_path, f"Image {image_id} - Input"),
        (gradcam_overlay_path, f"Image {image_id} - Grad-CAM"),
        (saliency_overlay_path, f"Image {image_id} - Saliency"),
        (lime_path, f"Image {image_id} - LIME"),
        (shap_overlay_path, f"Image {image_id} - SHAP"),
    ]

    available = [(p, t) for p, t in image_paths if p is not None and os.path.exists(p)]
    n = len(available)
    cols = 3
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (path, title) in zip(axes, available):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(f"Image {image_id} XAI Explanation", fontsize=14)
    fig.text(0.02, 0.01, summary_text, ha="left", va="bottom", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.18, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)
def region_to_short_text(region):
    bbox = region.get("bbox", {})
    return (
        f"x={bbox.get('x')}, y={bbox.get('y')}, "
        f"w={bbox.get('width')}, h={bbox.get('height')}, "
        f"score={region.get('score', 'unknown')}"
    )


def build_template_explanation(image_item):
    score = float(image_item.get("discriminator_score", 0.0))
    source = image_item.get("source", "unknown")

    gradcam_regions = image_item.get("gradcam", {}).get("important_regions", [])
    saliency_regions = image_item.get("saliency", {}).get("important_regions", [])
    lime_regions = image_item.get("lime", {}).get("important_regions", [])
    shap_regions = image_item.get("shap", {}).get("important_regions", [])

    method_agreement = image_item.get("method_agreement", {})
    avg_iou = method_agreement.get("average_iou", 0.0)
    agreement_level = method_agreement.get("agreement_level", "low")
    xai_descriptions = get_xai_method_descriptions()

    if score >= 0.7:
        score_text = "The discriminator gave this image a high realness score, so it strongly considered the image to be real."
    elif score >= 0.4:
        score_text = "The discriminator gave this image a moderate score, so it considered the image somewhat realistic but not fully convincing."
    else:
        score_text = "The discriminator gave this image a very low score, so it considered the image unlikely to be real."

    gradcam_text = (
        f"- Grad-CAM: {xai_descriptions['Grad-CAM']} It highlighted {len(gradcam_regions)} region(s). The top region is at {region_to_short_text(gradcam_regions[0])}."
        if gradcam_regions else
        f"- Grad-CAM: {xai_descriptions['Grad-CAM']} No strong region was detected."
    )
    saliency_text = (
        f"- Saliency: {xai_descriptions['Saliency']} It highlighted {len(saliency_regions)} region(s). The top region is at {region_to_short_text(saliency_regions[0])}."
        if saliency_regions else
        f"- Saliency: {xai_descriptions['Saliency']} No strong region was detected."
    )
    lime_text = (
        f"- LIME: {xai_descriptions['LIME']} It highlighted {len(lime_regions)} region(s). The top region is at {region_to_short_text(lime_regions[0])}."
        if lime_regions else
        f"- LIME: {xai_descriptions['LIME']} No strong region was detected."
    )
    shap_text = (
        f"- SHAP: {xai_descriptions['SHAP']} It highlighted {len(shap_regions)} region(s). The top region is at {region_to_short_text(shap_regions[0])}."
        if shap_regions else
        f"- SHAP: {xai_descriptions['SHAP']} No strong region was detected."
    )

    agreement_text = (
        f"The explanation methods show {agreement_level} agreement with an average IoU of {avg_iou}. "
        f"This means the methods {'focus on similar regions' if agreement_level == 'high' else 'do not strongly overlap and may be capturing different cues'}."
    )

    interpretation_text = (
        f"Because this is a {source} image, the explanation should be read as evidence of which visual regions influenced the discriminator most. "
        f"The combination of Grad-CAM, Saliency, LIME, and SHAP suggests the discriminator relied on a mixture of broad structure and local details rather than a single perfectly shared region."
    )

    return (
        "Decision summary:\n"
        f"{score_text}\n\n"
        "Method-wise explanation:\n"
        f"{gradcam_text}\n"
        f"{saliency_text}\n"
        f"{lime_text}\n"
        f"{shap_text}\n\n"
        "Agreement analysis:\n"
        f"{agreement_text}\n\n"
        "Interpretation:\n"
        f"{interpretation_text}"
    )


def slm_output_is_bad(text):
    if not text or len(text.strip()) < 60:
        return True

    required_sections = [
        "Decision summary:",
        "Method-wise",
        "Agreement analysis:",
        "Interpretation:",
    ]
    if not all(section in text for section in required_sections):
        return True

    weird_symbol_count = sum(text.count(ch) for ch in ['#', '$', '%', '&'])
    exclam_count = text.count('!')
    if weird_symbol_count >= 5 or exclam_count >= 12:
        return True

    bad_patterns = [
        "region 4",
        "region 5",
        "region 6",
        "saliencymethod",
        "shappicked",
        "grad-camp",
        "saliene",
    ]
    lowered = text.lower()
    if any(p in lowered for p in bad_patterns):
        return True

    return False


def draw_regions_on_overlay_rgb(overlay_rgb, regions, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on an RGB overlay image using region bbox entries."""
    if overlay_rgb.dtype != np.uint8:
        overlay_rgb = np.clip(overlay_rgb * 255, 0, 255).astype("uint8")

    img_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    for region in regions:
        bbox = region["bbox"]
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_overlay_with_boxes(save_path, overlay_rgb, regions):
    boxed = draw_regions_on_overlay_rgb(overlay_rgb, regions)
    cv2.imwrite(save_path, cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))
    return save_path


# ------------------ Region extraction helpers for overlays ------------------

def normalize_map_for_regions(xai_map):
    if isinstance(xai_map, torch.Tensor):
        arr = xai_map.detach().float().cpu().numpy()
    else:
        arr = np.asarray(xai_map)

    arr = np.squeeze(arr)
    arr = np.abs(arr).astype(np.float32)

    if arr.ndim == 3:
        arr = np.mean(arr, axis=0) if arr.shape[0] <= 4 else np.mean(arr, axis=2)

    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / (max_val + 1e-8)
    return arr


def extract_regions_with_fallback(xai_map, image_size, threshold=0.6, min_area=20):
    base_map = normalize_map_for_regions(xai_map)

    # Resize to the same resolution used for overlays so region boxes align visually.
    resized_map = cv2.resize(base_map, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    candidate_thresholds = [threshold, 0.45, 0.30, 0.20, 0.10]
    candidate_min_areas = [min_area, max(8, min_area // 2), 5]

    best_regions = []
    for thr in candidate_thresholds:
        for area in candidate_min_areas:
            regions = extract_regions_from_map(resized_map, threshold=thr, min_area=area)
            if len(regions) > 0:
                return regions
            best_regions = regions

    return best_regions


def save_comparison_panel(save_path, input_image_path, gradcam_path, saliency_path, lime_path, image_id, source):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    panels = [
        (input_image_path, f"Image {image_id} - Input ({source})"),
        (gradcam_path, f"Image {image_id} - Grad-CAM"),
        (saliency_path, f"Image {image_id} - Saliency"),
        (lime_path, f"Image {image_id} - LIME"),
    ]

    for ax, (path, title) in zip(axes, panels):
        if path is not None and os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------ SLM explanation helpers ------------------


def format_regions_for_slm(method_name, regions):
    if not regions:
        return f"{method_name}: no important regions detected."

    lines = [f"{method_name}:"]
    for i, region in enumerate(regions, start=1):
        bbox = region.get("bbox", {})
        centroid = region.get("centroid", {})
        area = region.get("area", "unknown")
        score = region.get("score", "unknown")
        lines.append(
            f"  Region {i}: "
            f"bbox=(x={bbox.get('x')}, y={bbox.get('y')}, w={bbox.get('width')}, h={bbox.get('height')}), "
            f"centroid=({centroid.get('x')}, {centroid.get('y')}), "
            f"area={area}, score={score}"
        )
    return "\n".join(lines)

# --- Helper for XAI method descriptions for SLM explanations ---
def get_xai_method_descriptions():
    return {
        "Grad-CAM": "Checks which broader high-level regions in the image most influenced the discriminator's real or fake decision.",
        "Saliency": "Checks which exact pixels or very local details the discriminator is most sensitive to.",
        "LIME": "Checks which interpretable image patches or segments change the discriminator decision most when perturbed.",
        "SHAP": "Checks how much each region contributes to pushing the discriminator output toward real or fake compared with a baseline."
    }


def build_slm_prompt(image_item):
    image_id = image_item.get("image_id", "unknown")
    source = image_item.get("source", "unknown")
    discriminator_score = image_item.get("discriminator_score", "unknown")

    gradcam_regions = image_item.get("gradcam", {}).get("important_regions", [])
    saliency_regions = image_item.get("saliency", {}).get("important_regions", [])
    lime_regions = image_item.get("lime", {}).get("important_regions", [])
    shap_regions = image_item.get("shap", {}).get("important_regions", [])

    method_agreement = image_item.get("method_agreement", {})
    avg_iou = method_agreement.get("average_iou", None)
    agreement_level = method_agreement.get("agreement_level", None)
    xai_descriptions = get_xai_method_descriptions()

    system_instruction = (
        "You explain GAN discriminator evidence from XAI outputs. "
        "Use only the structured evidence provided. "
        "Do not invent scene details that are not present in the evidence. "
        "Write plain, clean English with no markdown, no special symbols, and no image tags."
    )

    user_content = f"""
Image evidence:
- image_id: {image_id}
- source: {source}
- discriminator_score: {discriminator_score}

{format_regions_for_slm("Grad-CAM", gradcam_regions)}

{format_regions_for_slm("Saliency", saliency_regions)}

{format_regions_for_slm("LIME", lime_regions)}

{format_regions_for_slm("SHAP", shap_regions)}

Method agreement:
- agreement_level: {agreement_level}
- average_iou: {avg_iou}

What each XAI method checks:
- Grad-CAM: {xai_descriptions['Grad-CAM']}
- Saliency: {xai_descriptions['Saliency']}
- LIME: {xai_descriptions['LIME']}
- SHAP: {xai_descriptions['SHAP']}

Write the answer using exactly these section titles and nothing else:
Decision summary:
Method-wise explanation:
Agreement analysis:
Interpretation:

Requirements:
- 2 to 3 sentences for Decision summary.
- One bullet line each for Grad-CAM, Saliency, LIME, and SHAP under Method-wise explanation.
- In Method-wise explanation, state both what the method generally checks and what it highlighted in this specific image.
- 1 to 3 sentences for Agreement analysis.
- 2 to 4 sentences for Interpretation.
- No markdown image syntax.
- No repeated punctuation.
- No decorative symbols.
""".strip()

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def cleanup_slm_text(text):
    # Remove markdown image leftovers and obvious artifacts
    text = text.replace("![](", "").replace("```", "")

    # Join words that were split by repeated exclamation marks, e.g. jud!ged -> judged
    text = re.sub(r'(?<=[A-Za-z])!+(?=[A-Za-z])', '', text)

    # Turn punctuation wrapped in ! into normal punctuation
    text = re.sub(r'!+([:#.,;\-])', r'\1', text)
    text = re.sub(r'([:#.,;\-])!+', r'\1', text)

    # Convert bullet markers like *! or !* into a simple dash bullet
    text = re.sub(r'\s*[!*]+\s*', ' ', text)

    # Normalize section titles that may be broken by symbols
    text = re.sub(r'Decision\s*summary', 'Decision summary', text, flags=re.IGNORECASE)
    text = re.sub(r'Method\s*wise\s*explanation', 'Method-wise explanation', text, flags=re.IGNORECASE)
    text = re.sub(r'Agreement\s*analysis', 'Agreement analysis', text, flags=re.IGNORECASE)
    text = re.sub(r'Interpretation', 'Interpretation', text, flags=re.IGNORECASE)

    # Remove repeated spaces and repeated punctuation
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\.{2,}', '.', text)

    # Ensure bullet lines under method-wise explanation stay readable
    text = re.sub(r'\bGrad-CAM\b\s*', 'Grad-CAM: ', text)
    text = re.sub(r'\bSaliency\b\s*', 'Saliency: ', text)
    text = re.sub(r'\bLIME\b\s*', 'LIME: ', text)
    text = re.sub(r'\bSHAP\b\s*', 'SHAP: ', text)

    # Final cleanup around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_slm_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    preferred_device = device
    dtype = torch.float16 if preferred_device in ["cuda", "mps"] else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model.to(preferred_device)
        actual_device = preferred_device
    except Exception as e:
        print("Initial SLM loading on preferred device failed. Falling back to CPU.")
        print(e)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model.to("cpu")
        actual_device = "cpu"
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    model.eval()
    return tokenizer, model, actual_device


@torch.no_grad()
def generate_slm_text(prompt_messages, tokenizer, model, device, max_new_tokens=220):
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    try:
        output_ids = model.generate(**inputs, **generation_kwargs)
        actual_device = device
    except Exception as e:
        if device != "cpu":
            print("SLM generation failed on preferred device. Retrying on CPU.")
            print(e)
            model = model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            output_ids = model.generate(**inputs, **generation_kwargs)
            actual_device = "cpu"
        else:
            raise

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    text = cleanup_slm_text(text)
    return text, actual_device, model


# ------------------ Method agreement helpers ------------------

def bbox_iou(box_a, box_b):
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["width"], ay1 + box_a["height"]

    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["width"], by1 + box_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def compute_method_agreement(gradcam_regions, saliency_regions, lime_regions, shap_regions):
    methods = {
        "gradcam": gradcam_regions,
        "saliency": saliency_regions,
        "lime": lime_regions,
        "shap": shap_regions,
    }

    method_names = list(methods.keys())
    pairwise = []

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            m1 = method_names[i]
            m2 = method_names[j]
            r1 = methods[m1]
            r2 = methods[m2]

            if len(r1) == 0 or len(r2) == 0:
                pairwise.append({
                    "method_a": m1,
                    "method_b": m2,
                    "top_region_iou": None,
                    "status": "missing-region"
                })
                continue

            iou = bbox_iou(r1[0]["bbox"], r2[0]["bbox"])
            pairwise.append({
                "method_a": m1,
                "method_b": m2,
                "top_region_iou": round(float(iou), 4),
                "status": "ok"
            })

    valid_ious = [x["top_region_iou"] for x in pairwise if x["top_region_iou"] is not None]
    average_iou = round(float(sum(valid_ious) / len(valid_ious)), 4) if valid_ious else 0.0

    if average_iou >= 0.35:
        agreement_level = "high"
    elif average_iou >= 0.15:
        agreement_level = "medium"
    else:
        agreement_level = "low"

    return {
        "pairwise_top_region_iou": pairwise,
        "average_iou": average_iou,
        "agreement_level": agreement_level
    }


def save_lime_output(lime_result, save_path):
    if isinstance(lime_result, dict):
        lime_img = lime_result.get("image")
    else:
        lime_img = lime_result

    if isinstance(lime_img, np.ndarray):
        if lime_img.dtype != np.uint8:
            lime_img = np.clip(lime_img, 0, 1)
            lime_img = (lime_img * 255).astype("uint8")

        if lime_img.ndim == 3 and lime_img.shape[2] == 3:
            cv2.imwrite(save_path, cv2.cvtColor(lime_img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(save_path, lime_img)
        return save_path

    return None


def collect_real_images(test_loader, num_images, device):
    collected = []
    total = 0
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        collected.append(imgs)
        total += imgs.size(0)
        if total >= num_images:
            break

    if not collected:
        raise ValueError("No real images could be collected from the validation/test loader.")

    return torch.cat(collected, dim=0)[:num_images]


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    args.epochs = 1
    args.run_dir = os.path.join("experiments", args.run_name)
    os.makedirs(args.run_dir, exist_ok=True)

    _, test_loader = get_data_loader(args, target_class=args.target_class)

    model = DCGAN_MODEL(
        args,
        use_xai=False,
        lambda_xai=0.0,
        xai_mode=None,
    )

    print("Loading trained model...")
    model.load_model(args.load_G, args.load_D)
    model.G.eval()
    model.D.eval()

    slm_tokenizer = None
    slm_model = None
    slm_device = None
    if args.generate_slm_explanations:
        requested_slm_device = "cpu" if args.force_slm_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading SLM for automatic explanations on: {requested_slm_device}")
        slm_tokenizer, slm_model, slm_device = load_slm_and_tokenizer(args.slm_model, requested_slm_device)
        print(f"SLM active device: {slm_device}")

    report_dir = os.path.join(args.run_dir, f"xai_report_{args.source}")
    os.makedirs(report_dir, exist_ok=True)

    if args.source == "generated":
        with torch.no_grad():
            z = torch.randn(args.num_images, 100, 1, 1, device=device)
            images = model.G(z)
    else:
        images = collect_real_images(test_loader, args.num_images, device)

    gradcam_explainer = GradCAM(model.D, model.D.features[11])
    saliency_explainer = SaliencyMap(model.D)

    gradcam_maps = gradcam_explainer.generate(images)
    saliency_maps = saliency_explainer.generate(images)

    with torch.no_grad():
        discriminator_scores = model.D(images).detach().cpu().tolist()

    summary = {
        "run_name": args.run_name,
        "source": args.source,
        "num_images": args.num_images,
        "images": [],
    }

    print("Applying XAI...")
    for idx in range(args.num_images):
        image_dir = os.path.join(report_dir, f"image_{idx + 1:03d}")
        os.makedirs(image_dir, exist_ok=True)

        image = images[idx]
        image_norm = (image + 1) / 2
        image_vis = F.interpolate(
            image_norm.unsqueeze(0),
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        input_path = os.path.join(image_dir, "input_image.png")
        utils.save_image(image_vis, input_path)

        # Grad-CAM
        gradcam_map = gradcam_maps[idx]
        gradcam_overlay = overlay_cam(image, gradcam_map)
        gradcam_overlay = cv2.resize(gradcam_overlay, (args.image_size, args.image_size))
        gradcam_overlay_path = os.path.join(image_dir, "gradcam_overlay.png")
        cv2.imwrite(
            gradcam_overlay_path,
            cv2.cvtColor((gradcam_overlay * 255).astype("uint8"), cv2.COLOR_RGB2BGR),
        )
        gradcam_regions = extract_regions_with_fallback(
            gradcam_map,
            args.image_size,
            threshold=args.threshold,
            min_area=args.min_area,
        )
        gradcam_overlay_boxes_path = os.path.join(image_dir, "gradcam_overlay_boxes.png")
        save_overlay_with_boxes(gradcam_overlay_boxes_path, gradcam_overlay, gradcam_regions)

        # Saliency
        saliency_map = saliency_maps[idx]
        saliency_overlay = overlay_saliency(image, saliency_map)
        saliency_overlay = cv2.resize(saliency_overlay, (args.image_size, args.image_size))
        saliency_overlay_path = os.path.join(image_dir, "saliency_overlay.png")
        cv2.imwrite(
            saliency_overlay_path,
            cv2.cvtColor((saliency_overlay * 255).astype("uint8"), cv2.COLOR_RGB2BGR),
        )
        saliency_regions = extract_regions_with_fallback(
            saliency_map,
            args.image_size,
            threshold=args.threshold,
            min_area=args.min_area,
        )
        saliency_overlay_boxes_path = os.path.join(image_dir, "saliency_overlay_boxes.png")
        save_overlay_with_boxes(saliency_overlay_boxes_path, saliency_overlay, saliency_regions)

        # LIME
        lime_result = run_lime(model, image, min_area=args.min_area, top_k=5)
        lime_path = os.path.join(image_dir, "lime_explanation.png")
        lime_output_path = save_lime_output(lime_result, lime_path)
        lime_regions = lime_result.get("important_regions", []) if isinstance(lime_result, dict) else []
        lime_weights_path = os.path.join(image_dir, "lime_weights.json")
        if isinstance(lime_result, dict):
            with open(lime_weights_path, "w") as f:
                json.dump(lime_result.get("raw_weights", {}), f, indent=2)
        else:
            lime_weights_path = None

        lime_overlay_boxes_path = os.path.join(image_dir, "lime_overlay_boxes.png")
        if lime_output_path is not None and os.path.exists(lime_output_path):
            lime_img_bgr = cv2.imread(lime_output_path)
            lime_img_rgb = cv2.cvtColor(lime_img_bgr, cv2.COLOR_BGR2RGB)
            save_overlay_with_boxes(lime_overlay_boxes_path, lime_img_rgb, lime_regions)
        else:
            lime_overlay_boxes_path = None

        # SHAP
        with torch.no_grad():
            bg_z = torch.randn(10, 100, 1, 1, device=device)
            bg_images = model.G(bg_z)

        background = bg_images.permute(0, 2, 3, 1).detach().cpu().numpy()
        test_img = images[idx:idx + 1].permute(0, 2, 3, 1).detach().cpu().numpy()

        shap_result = run_shap(model, background, test_img, min_area=args.min_area, top_k=5)
        shap_path = os.path.join(image_dir, "shap_explanation.npy")
        np.save(shap_path, shap_result["shap_values"], allow_pickle=True)
        shap_regions_path = os.path.join(image_dir, "shap_regions.json")
        shap_overlay_boxes_path = None
        shap_map = shap_result.get("shap_map")
        if shap_map is not None:
            shap_regions = extract_regions_with_fallback(
                shap_map,
                args.image_size,
                threshold=args.threshold,
                min_area=args.min_area,
            )

            shap_vis = normalize_map_for_regions(shap_map)
            shap_vis = cv2.resize(shap_vis, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
            shap_vis = (shap_vis * 255).astype("uint8")
            shap_vis = cv2.applyColorMap(shap_vis, cv2.COLORMAP_JET)
            shap_vis = cv2.cvtColor(shap_vis, cv2.COLOR_BGR2RGB)
            shap_overlay_boxes_path = os.path.join(image_dir, "shap_overlay_boxes.png")
            save_overlay_with_boxes(shap_overlay_boxes_path, shap_vis, shap_regions)

        with open(shap_regions_path, "w") as f:
            json.dump(shap_regions, f, indent=2)

        method_agreement = compute_method_agreement(
            gradcam_regions,
            saliency_regions,
            lime_regions,
            shap_regions,
        )

        score = float(discriminator_scores[idx])

        print(f"\nImage {idx + 1} analysis")
        print("---------------------------------")
        print(f"Source: {args.source}")
        print(f"Discriminator score: {score:.4f}")
        print(f"Grad-CAM regions: {len(gradcam_regions)}")
        print(f"Saliency regions: {len(saliency_regions)}")
        print(f"LIME regions: {len(lime_regions)}")
        print(f"SHAP regions: {len(shap_regions)}")
        print(f"Method agreement: {method_agreement['agreement_level']} (avg IoU = {method_agreement['average_iou']})")

        comparison_panel_path = os.path.join(image_dir, "xai_comparison_panel.png")
        save_comparison_panel(
            comparison_panel_path,
            input_path,
            gradcam_overlay_boxes_path,
            saliency_overlay_boxes_path,
            lime_overlay_boxes_path if lime_overlay_boxes_path is not None else lime_output_path,
            idx + 1,
            args.source,
        )


        image_report = {
            "image_id": idx + 1,
            "source": args.source,
            "discriminator_score": round(score, 4),
            "gradcam": {
                "important_regions": gradcam_regions,
                "overlay_path": gradcam_overlay_path,
                "overlay_boxes_path": gradcam_overlay_boxes_path,
            },
            "saliency": {
                "important_regions": saliency_regions,
                "overlay_path": saliency_overlay_path,
                "overlay_boxes_path": saliency_overlay_boxes_path,
            },
            "lime": {
                "important_regions": lime_regions,
                "output_path": lime_output_path,
                "weights_path": lime_weights_path,
                "overlay_boxes_path": lime_overlay_boxes_path,
            },
            "shap": {
                "important_regions": shap_regions,
                "output_path": shap_path,
                "regions_path": shap_regions_path,
                "overlay_boxes_path": shap_overlay_boxes_path,
            },
            "method_agreement": method_agreement,
            "comparison_panel_path": comparison_panel_path,
        }

        if args.generate_slm_explanations:
            slm_messages = build_slm_prompt(image_report)
            slm_explanation, slm_device, slm_model = generate_slm_text(
                slm_messages,
                slm_tokenizer,
                slm_model,
                slm_device,
                max_new_tokens=args.slm_max_new_tokens,
            )

            if slm_output_is_bad(slm_explanation):
                print("SLM output looked malformed. Replacing it with a clean template explanation.")
                slm_explanation = build_template_explanation(image_report)

            image_report["slm_explanation"] = slm_explanation
            image_report["slm_generation_device"] = slm_device

            slm_explanation_path = os.path.join(image_dir, "slm_explanation.txt")
            with open(slm_explanation_path, "w", encoding="utf-8") as f:
                f.write(slm_explanation)
            image_report["slm_explanation_path"] = slm_explanation_path

            display_panel_path = os.path.join(image_dir, "xai_explanation_display.png")
            image_report["xai_explanation_display_path"] = display_panel_path

            print("SLM explanation generated.")
            print(slm_explanation)

            if args.show_images:
                display_gradcam_result(
                    input_path,
                    gradcam_overlay_boxes_path,
                    slm_explanation,
                    idx + 1,
                    saliency_overlay_boxes_path,
                    lime_overlay_boxes_path if lime_overlay_boxes_path is not None else lime_output_path,
                    shap_overlay_boxes_path,
                    save_path=display_panel_path,
                    show_plot=True,
                )
            else:
                display_gradcam_result(
                    input_path,
                    gradcam_overlay_boxes_path,
                    slm_explanation,
                    idx + 1,
                    saliency_overlay_boxes_path,
                    lime_overlay_boxes_path if lime_overlay_boxes_path is not None else lime_output_path,
                    shap_overlay_boxes_path,
                    save_path=display_panel_path,
                    show_plot=False,
                )

        elif args.show_images:
            preview_text = (
                f"Discriminator score: {score:.4f}\n"
                f"Method agreement: {method_agreement['agreement_level']} "
                f"(avg IoU = {method_agreement['average_iou']})"
            )
            display_panel_path = os.path.join(image_dir, "xai_explanation_display.png")
            image_report["xai_explanation_display_path"] = display_panel_path
            display_gradcam_result(
                input_path,
                gradcam_overlay_boxes_path,
                preview_text,
                idx + 1,
                saliency_overlay_boxes_path,
                lime_overlay_boxes_path if lime_overlay_boxes_path is not None else lime_output_path,
                shap_overlay_boxes_path,
                save_path=display_panel_path,
                show_plot=True,
            )

        with open(os.path.join(image_dir, "single_image_xai.json"), "w") as f:
            json.dump(image_report, f, indent=2)

        summary["images"].append(image_report)

    valid_agreements = [img["method_agreement"]["average_iou"] for img in summary["images"]]
    avg_agreement = round(float(sum(valid_agreements) / len(valid_agreements)), 4) if valid_agreements else 0.0
    summary["agreement_summary"] = {
        "num_images": len(summary["images"]),
        "average_iou_across_images": avg_agreement,
        "agreement_levels": [img["method_agreement"]["agreement_level"] for img in summary["images"]]
    }

    summary_path = os.path.join(report_dir, "xai_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.generate_slm_explanations:
        print("Automatic SLM explanations were generated for all processed images.")
    print(f"Average method agreement across images: {summary['agreement_summary']['average_iou_across_images']}")
    print("Done.")
    print(f"All {args.source} XAI outputs saved in: {report_dir}")
    print(f"Combined JSON saved at: {summary_path}")


if __name__ == "__main__":
    main()
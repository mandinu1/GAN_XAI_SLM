import json
import os
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INPUT_JSON = "experiments/new/dcgan_mountain_256/xai_report_generated/image_001/single_image_xai.json"
OUTPUT_JSON = "experiments/new/dcgan_mountain_256/xai_report_generated/image_001/explanations.json"
OUTPUT_TXT = "experiments/new/dcgan_mountain_256/xai_report_generated/image_001/explanations.txt"
MAX_NEW_TOKENS = 220


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_image_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "images" in data and isinstance(data["images"], list):
        return data["images"]
    return [data]


def format_regions(method_name: str, regions: List[Dict[str, Any]]) -> str:
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


def build_prompt(image_item: Dict[str, Any]) -> str:
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

    system_instruction = (
        "You are helping explain GAN discriminator evidence from XAI outputs. "
        "Write a clear, accurate explanation based only on the provided structured evidence. "
        "Do not invent image content that is not supported by the data. "
        "Explain what the discriminator score suggests, what each XAI method highlights, "
        "whether the methods agree, and what this may imply about the discriminator's reasoning."
    )

    user_content = f"""
Image evidence:
- image_id: {image_id}
- source: {source}
- discriminator_score: {discriminator_score}

{format_regions("Grad-CAM", gradcam_regions)}

{format_regions("Saliency", saliency_regions)}

{format_regions("LIME", lime_regions)}

{format_regions("SHAP", shap_regions)}

Method agreement:
- agreement_level: {agreement_level}
- average_iou: {avg_iou}

Write the output in this format:

Decision summary:
<2-3 sentences>

Method-wise explanation:
- Grad-CAM: ...
- Saliency: ...
- LIME: ...
- SHAP: ...

Agreement analysis:
<1-3 sentences>

Interpretation:
<2-4 sentences>
""".strip()

    return (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model_and_tokenizer(model_name: str, device: str) -> Tuple[Any, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen 7B on MPS can be unstable. We try MPS first, then safely fall back.
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
        print("Initial model loading on preferred device failed. Falling back to CPU.")
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

    model.eval()
    return tokenizer, model, actual_device


@torch.no_grad()
def generate_text(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 220,
) -> Tuple[str, str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        actual_device = device
    except Exception as e:
        if device != "cpu":
            print("Generation failed on preferred device. Retrying on CPU.")
            print(e)
            model = model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
            actual_device = "cpu"
        else:
            raise

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, actual_device, model


def save_outputs(results: List[Dict[str, Any]], output_json: str, output_txt: str) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_txt, "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"Image ID: {item['image_id']}\n")
            f.write(f"Source: {item['source']}\n")
            f.write(f"Generation device used: {item['generation_device']}\n")
            f.write(item["explanation"])
            f.write("\n" + "=" * 80 + "\n\n")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    data = load_json(INPUT_JSON)
    image_items = ensure_image_list(data)

    tokenizer, model, active_device = load_model_and_tokenizer(MODEL_NAME, device)
    print(f"Model active device: {active_device}")

    results = []
    for idx, image_item in enumerate(image_items, start=1):
        prompt = build_prompt(image_item)
        explanation, active_device, model = generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=active_device,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        result = {
            "image_id": image_item.get("image_id"),
            "source": image_item.get("source"),
            "discriminator_score": image_item.get("discriminator_score"),
            "generation_device": active_device,
            "explanation": explanation,
        }
        results.append(result)

        print(f"\n[{idx}/{len(image_items)}] Image {result['image_id']} done.\n")
        print(explanation)
        print("\n" + "-" * 80)

    save_outputs(results, OUTPUT_JSON, OUTPUT_TXT)
    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved text: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
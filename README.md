
# GAN_XAI_SLM

Explainable GAN Discriminator Analysis with Small Language Models

This repository contains the implementation of a research project that investigates the **decision-making behavior of GAN discriminators** using **Explainable AI (XAI)** techniques and **Small Language Models (SLMs)**.

The system combines:

• **DCGAN image generation**  
• **XAI-guided training (Grad-CAM + Saliency losses)**  
• **Multi-method explanation (Grad-CAM, Saliency, LIME, SHAP)**  
• **Region extraction and agreement analysis**  
• **Natural language explanation generation using SLMs**

---

# Research Overview

The goal of this project is to understand **what features a GAN discriminator learns** during training.

The proposed framework includes:

1. **GAN Training**
2. **XAI-Guided Training**
3. **Multi-Method Explainability**
4. **Region Extraction and Agreement Analysis**
5. **SLM-based Explanation Generation**

Pipeline:

Dataset → DCGAN Training → Generated Images
↓
Discriminator Prediction
↓
XAI Methods
(Grad-CAM, Saliency, LIME, SHAP)
↓
Important Region Extraction
↓
IoU Agreement Analysis
↓
Structured JSON Evidence
↓
SLM Explanation Generation
↓
Visual + Text Explanation Output

---

# Repository Structure

```

GAN_XAI_SLM/
│
├── models/
│   ├── dcgan.py
│   ├── gan.py
│
├── Xai_tools/
│   ├── grad_cam.py
│   ├── saliency_map.py
│   ├── lime_explainer.py
│   ├── shap_explainer.py
│
├── utils/
│   ├── data_loader.py
│   ├── FID.py
│   ├── inception_score.py
│   ├── msssim.py
│
├── experiments/
│
├── main.py
├── GAN_XAI.py
├── GAN_XAI_slm.ipynb
│
├── requirements.txt
└── README.md



Running the Project

1. Train Baseline DCGAN

python main.py \
  --model DCGAN \
  --dataset places365 \
  --dataroot ./places365 \
  --epochs 2000 \
  --batch_size 64 \
  --channels 3 \
  --target_class mountain \
  --run_name experiments/dcgan_mountain_256 \
  --train


2. Train with XAI-Guided Training

Grad-CAM guided training:

python main.py \
  --model DCGAN \
  --dataset places365 \
  --dataroot ./places365 \
  --epochs 2000 \
  --batch_size 64 \
  --channels 3 \
  --target_class mountain \
  --run_name experiments/dcgan_gradcam \
  --train \
  --use_xai \
  --xai_mode gradcam \
  --lambda_xai 0.1

Saliency guided training:

--xai_mode saliency

Both:

--xai_mode both


⸻

3. Run XAI Analysis

After training, explanations can be generated.

Example command:

python GAN_XAI.py \
  --dataset places365 \
  --dataroot ./places365 \
  --channels 3 \
  --target_class mountain \
  --run_name experiments/dcgan_mountain_256 \
  --load_G experiments/dcgan_mountain_256/generator_DCGAN_best_fid.pth \
  --load_D experiments/dcgan_mountain_256/discriminator_DCGAN_best_fid.pth \
  --source real \
  --num_images 1 \
  --image_size 256 \
  --show_images

Outputs include:

original image
Grad-CAM heatmap
Saliency map
LIME explanation
SHAP explanation
bounding boxes


⸻

4. Generate SLM Explanations

The system converts structured XAI evidence into natural language explanations.

Example:

python GAN_XAI.py \
  --dataset places365 \
  --dataroot ./places365 \
  --channels 3 \
  --target_class mountain \
  --run_name experiments/dcgan_mountain_256 \
  --load_G experiments/dcgan_mountain_256/generator_DCGAN_best_fid.pth \
  --load_D experiments/dcgan_mountain_256/discriminator_DCGAN_best_fid.pth \
  --source real \
  --num_images 1 \
  --image_size 256 \
  --generate_slm_explanations \
  --slm_model Qwen/Qwen2.5-3B-Instruct

Supported SLM models:

Qwen/Qwen2.5-3B-Instruct
microsoft/Phi-3.5-mini-instruct
mistralai/Mistral-7B-Instruct


⸻

Evaluation Metrics

The project evaluates GAN performance using:

FID (Fréchet Inception Distance)

Lower is better.

Measures similarity between generated and real distributions.

⸻

Inception Score (IS)

Higher is better.

Measures image quality and diversity.

⸻

MS-SSIM

Lower is better.

Measures diversity of generated images.

⸻



The SLM generates structured explanations containing:
	•	Decision Summary
	•	Method-wise Explanation
	•	Agreement Analysis
	•	Interpretation



Research Contribution

This project proposes a unified framework combining:

XAI-Guided GAN Training

Grad-CAM loss
Saliency loss

Multi-method Explainability

Grad-CAM
Saliency
LIME
SHAP

Natural Language Model Explanations

Qwen-3B
Phi-3.5
Mistral-7B




# GAN_XAI_SLM

A research project that combines **DCGAN**, **Explainable AI (XAI)**, and **Small Language Models (SLMs)** to analyze how a GAN discriminator makes real/fake decisions and to generate natural-language explanations from structured XAI evidence.

## Overview

This project studies the internal reasoning of a GAN discriminator by:

- training a **DCGAN** on the **Places365 Mountain** class
- optionally using **XAI-guided training losses**
  - Grad-CAM attention loss
  - Saliency attention loss
- applying multiple XAI methods to trained models
  - Grad-CAM
  - Saliency Maps
  - LIME
  - SHAP
- extracting important regions and comparing method agreement using **IoU**
- converting structured explanation outputs into text using **SLMs**
  - Qwen2.5-3B-Instruct
  - Phi-3.5 Mini
  - Mistral-7B-Instruct

## Project Structure

```text
GAN_XAI_SLM/
│
├── models/
│   ├── dcgan.py
│   └── ...
│
├── Xai_tools/
│   ├── grad_cam.py
│   ├── saliency_map.py
│   ├── lime_explainer.py
│   └── shap_explainer.py
│
├── utils/
│   ├── data_loader.py
│   ├── FID.py
│   ├── inception_score.py
│   ├── msssim.py
│   └── ...
│
├── main.py
├── GAN_XAI.py
├── GAN_XAI_slm.ipynb
├── requirements.txt
└── README.md

# Project Report: MediScan — Advancing Explainable AI in Medical Diagnostics

## 1. Abstract
The "MediScan" project addresses the critical need for transparency in AI-assisted medical diagnostics. While deep learning models achieve high accuracy in image classification, their "black box" nature often limits clinical trust. This project implements an Explainable AI (XAI) framework using Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize neural network attention on Chest X-rays and Skin Lesions. The result is a premium, clinician-centric dashboard that bridges the gap between complex AI logic and actionable medical insights.

## 2. Introduction
In modern healthcare, Computer Vision (CV) has become a secondary set of eyes for radiologists and dermatologists. However, a simple classification (e.g., "Pneumonia: 90%") is insufficient for diagnosis. Clinicians need to know *why* the AI reached that conclusion. MediScan provides this reasoning through real-time heatmap generation, pinpointing anatomical anomalies alongside traditional probability metrics.

## 3. Methodology

### 3.1 AI Engine Architecture
The system utilizes state-of-the-art convolutional neural networks (CNNs):
- **Pulmonary Analysis:** A DenseNet-121 architecture, pre-trained on large-scale medical datasets, is used for its superior feature extraction in low-contrast X-ray imagery.
- **Dermatological Analysis:** A ResNet-50 architecture is employed for skin lesion classification, providing a robust balance between depth and computational efficiency.

### 3.2 Explainability Framework (Grad-CAM)
We implemented Grad-CAM by hooking into the final convolutional layers of the models. By calculating the gradients of the target class with respect to the feature maps, we produce a coarse localization map highlighting the important regions in the image for prediction.

### 3.3 System Integration
The project follows a decoupled architecture:
- **Backend:** A Flask-based REST API handles image preprocessing, tensor transformation, and model inference.
- **Frontend:** A high-performance React application provides an interactive interface, featuring a real-time "Clinical Report" generator and PDF export capabilities.

## 4. Implementation Details
The user interface adheres to "Glassmorphism" design principles, utilizing subtle blurs and a clinical color palette (Medical Blue/Slate) to reduce cognitive load. A key innovation is the **Patient-Centric Report**, which contextualizes AI findings with user-inputted patient metadata (Name, Age, Gender), facilitating easier integration into clinical workflows.

## 5. Results and Discussion
Testing with public domain medical imagery demonstrated that the Grad-CAM heatmaps consistently aligned with visible pathological features, such as pulmonary opacities in X-rays or irregular pigment distributions in skin lesions. The system maintains sub-second inference times on consumer-grade hardware, making it suitable for real-time diagnostic assistance.

## 6. Conclusion
MediScan demonstrates that AI in healthcare can be both powerful and transparent. By prioritizing explainability and user experience, this capstone project provides a blueprint for next-generation clinical tools that empower rather than replace the medical professional.

---
**Domain:** Computer Vision / Healthcare AI
**Author:** BYOP Capstone Team
**Date:** March 2026

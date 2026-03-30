<div align="center">

# VELLORE INSTITUTE OF TECHNOLOGY

**School of Computer Science and Engineering**

---

## PROJECT REPORT

### On

# MediScan: Explainable AI for Medical Image Diagnostics

*Submitted in partial fulfilment of the requirements for the degree of*

## Bachelor of Technology
### in
## Artificial Intelligence

---

**Submitted by**

| | |
|---|---|
| **Name** | Hardik Verma |
| **Registration Number** | 23BAI10915 |
| **Programme** | B.Tech – Artificial Intelligence |
| **School** | School of Computer Science and Engineering |

---

*March 2026*

</div>

---

## Declaration

I, **Hardik Verma** (Reg. No. **23BAI10915**), hereby declare that the project report entitled **"MediScan: Explainable AI for Medical Image Diagnostics"** submitted to Vellore Institute of Technology in partial fulfilment of the requirements for the degree of Bachelor of Technology in Artificial Intelligence is an authentic record of work carried out by me.

The matter embodied in this report has not been submitted to any other university or institute for the award of any degree or diploma. All sources have been duly acknowledged.

&nbsp;

**Hardik Verma**  
Reg. No.: 23BAI10915  
Date: March 2026  
Place: Vellore

---

## Acknowledgement

I would like to express my sincere gratitude to everyone who supported me throughout this capstone project.

I thank the **School of Computer Science and Engineering, VIT** for providing the academic environment and resources that made this project possible.

I am grateful to the open-source communities behind **PyTorch**, **React**, and **Flask**, whose tools form the backbone of this system. The pioneering work of Rajpurkar et al. in CheXNet, and Selvaraju et al. in Grad-CAM, provided the foundational research that inspired the technical direction of this project.

Finally, I extend my thanks to my peers and family for their continued encouragement and feedback throughout the development process.

&nbsp;

**Hardik Verma**  
Reg. No.: 23BAI10915

---

**Domain:** Computer Vision / Healthcare AI  
**Student:** Hardik Verma | Reg. No.: 23BAI10915  
**Programme:** B.Tech – Artificial Intelligence, VIT  
**Academic Year:** 2025–2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Why This Problem Matters](#3-why-this-problem-matters)
4. [Approach & Methodology](#4-approach--methodology)
5. [System Architecture](#5-system-architecture)
6. [Key Technical Decisions](#6-key-technical-decisions)
7. [Implementation Details](#7-implementation-details)
8. [Challenges Faced](#8-challenges-faced)
9. [Results & Discussion](#9-results--discussion)
10. [What We Learned](#10-what-we-learned)
11. [Future Scope](#11-future-scope)
12. [Conclusion](#12-conclusion)

---

## 1. Abstract

MediScan is a web-based, AI-powered clinical diagnostic assistant that addresses a fundamental gap in healthcare AI: the lack of transparency in automated medical image analysis. While deep learning models can classify diseases with impressive accuracy, their inability to *explain* their reasoning has been a critical barrier to clinical adoption.

This project implements an **Explainable AI (XAI)** pipeline using **Gradient-weighted Class Activation Mapping (Grad-CAM)** applied on top of state-of-the-art convolutional neural networks (DenseNet-121 and ResNet-50). The result is a full-stack clinical tool that not only classifies conditions from Chest X-Rays and Skin Lesions, but visually highlights the exact anatomical regions that influenced the model's decision — turning a black-box prediction into a clinically interpretable insight.

---

## 2. Problem Statement

The central problem this project addresses is:

> **How can we make AI-driven medical image diagnosis both accurate *and* interpretable for clinical professionals?**

In current practice, a model might output:
> *"Pneumonia: 91% confidence"*

But a radiologist cannot act on this alone. They need to know:
- *Which region of the lung triggered this classification?*
- *Is the model focused on the right anatomical structure?*
- *Can this finding be cross-referenced with clinical history?*

Without answers to these questions, AI predictions remain unverifiable — and therefore, unusable in real clinical workflows.

MediScan directly addresses this gap by producing a **spatial attention heatmap** overlaid on the original image, alongside the prediction scores. This bridges the gap between model output and clinical utility.

---

## 3. Why This Problem Matters

### 3.1 Scale of Impact

Medical imaging is one of the highest-stakes domains for AI. According to the WHO, radiological examinations account for more than **3.6 billion medical imaging procedures** globally per year. Even a marginal improvement in diagnostic speed and accuracy at scale has enormous public health implications.

### 3.2 The Trust Gap in Clinical AI

A 2021 study in *Nature Medicine* found that a major barrier to AI adoption in hospitals is not accuracy, but **lack of interpretability**. Clinicians often distrust a system that cannot explain itself. A wrong prediction with a visible heatmap is more useful than a correct prediction with no reasoning — because the clinician can correct the former.

### 3.3 Accessibility in Under-Resourced Settings

Radiologists are scarce in many parts of the world. In India alone, there is approximately **1 radiologist per 100,000 people** — far below WHO recommendations. An AI-assisted diagnostic tool that is explainable, web-based, and runs locally (without cloud dependency) could serve as a first-pass screening tool in such settings.

### 3.4 Academic Relevance

This project sits at the intersection of three rapidly advancing fields:
- **Computer Vision** (image classification with CNNs)
- **Explainable AI / XAI** (making ML decisions transparent)
- **Healthcare AI** (applying ML to high-stakes clinical problems)

Mastering the intersection of these three areas is a highly valuable and sought-after skill in both industry and research.

---

## 4. Approach & Methodology

The solution was designed in three layers:

### 4.1 AI Layer — Classification + Explainability

Two pre-trained CNN models form the core AI engine:

| Mode | Model | Rationale |
|---|---|---|
| Chest X-Ray (Pulmonary) | **DenseNet-121** | Extensively benchmarked on ChestX-ray14 dataset; dense connections enable strong feature propagation in low-contrast imagery |
| Skin Lesion (Dermatological) | **ResNet-50** | Strong baseline for texture-based classification; well-suited for pigment variation in dermatological imagery |

Both models use **ImageNet pre-trained weights** via PyTorch's `torchvision` library. While these weights are not medically trained (a key limitation we acknowledge), they provide strong foundational feature extractors that demonstrate the *pipeline* concept convincingly.

**Explainability via Grad-CAM:**

Grad-CAM works by:
1. Running a forward pass through the network to get the prediction.
2. Backpropagating gradients of the target class score with respect to feature maps in the *last convolutional layer*.
3. Computing the mean of gradients across spatial dimensions (Global Average Pooling) to get per-channel importance weights.
4. Producing a weighted sum of feature activations, followed by a ReLU (to retain only positive contributions).
5. Upsampling the resulting coarse map to the original image size and overlaying it as a heatmap.

This produces a colour map (jet colormap via OpenCV) where **red/warm regions indicate high attention** and **blue/cool regions indicate low attention**.

### 4.2 Backend Layer — REST API

A **Flask** REST API serves as the bridge between the AI engine and the frontend. It exposes two endpoints (`/health`, `/analyze`) and handles:
- File ingestion (multipart form data)
- Routing to the correct AI model based on the requested `mode`
- Image preprocessing (resize to 224×224, normalize with ImageNet stats)
- Grad-CAM heatmap generation and OpenCV overlay
- Base64 encoding of the result image for transport
- Lazy model loading (models initialize on first request to save startup time)

### 4.3 Frontend Layer — Interactive Clinical Interface

A **React + Vite** single-page application provides the user experience. Key frontend decisions are discussed in Section 6, but the core flow is:

1. User selects diagnostic mode (X-Ray or Skin Lesion)
2. User enters optional patient metadata (name, age, gender)
3. User uploads an image (drag/drop or file picker)
4. Frontend POSTs to `/analyze`, displaying an animated loading state
5. On response, the app renders:
   - Side-by-side original image vs. Grad-CAM heatmap
   - Top-3 prediction scores with animated progress bars
   - A structured "Clinical Analysis Report" with patient metadata
   - A "Export PDF Report" action (using native `window.print()`)

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser / Client                    │
│   React + Vite + Tailwind CSS + Framer Motion           │
│   - Mode selector, Patient form, Image upload           │
│   - Animated loading states                             │
│   - Side-by-side image + heatmap viewer                 │
│   - Printable Clinical Report (window.print)            │
└─────────────────────────────┬───────────────────────────┘
                              │  POST /analyze
                              │  (multipart/form-data)
                              ▼
┌─────────────────────────────────────────────────────────┐
│                Flask REST API (Port 5000)               │
│   main.py                                               │
│   - /health : service status                            │
│   - /analyze : image intake, model routing              │
│   - Lazy model initialization                           │
│   - Base64 image serialization                          │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                   AI Engine (ai_engine.py)              │
│                                                         │
│   ┌─────────────────┐      ┌──────────────────────┐    │
│   │  MedicalModel   │      │       GradCAM         │    │
│   │  (xray mode)    │─────▶│  forward hook →       │    │
│   │  DenseNet-121   │      │  save_activation()    │    │
│   │  ImageNet weights│     │  backward hook →      │    │
│   └─────────────────┘      │  save_gradient()      │    │
│                             │  generate_heatmap()   │    │
│   ┌─────────────────┐      └──────────────────────┘    │
│   │  MedicalModel   │                │                  │
│   │  (skin mode)    │                ▼                  │
│   │  ResNet-50      │      ┌──────────────────────┐    │
│   │  ImageNet weights│     │  apply_heatmap()      │    │
│   └─────────────────┘      │  OpenCV JET colormap │    │
│                             │  → superimposed img  │    │
│                             └──────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Key Technical Decisions

### Decision 1: DenseNet-121 for Chest X-Rays (not ResNet)

**Why:** DenseNet-121 is the most widely used architecture in chest X-ray research (it is the backbone of the original CheXNet paper by Rajpurkar et al., 2017). Its dense connectivity pattern — where each layer receives feature maps from *all* preceding layers — allows fine-grained feature propagation that is especially beneficial for the subtle, low-contrast patterns in X-ray imagery (e.g., early-stage effusion or nodules). ResNet skips connections between every 2 layers; DenseNet connects every layer to every subsequent one.

**Alternative considered:** ResNet-50 for both modes (simpler implementation). Rejected because DenseNet-121 has established clinical literature backing its use for pulmonary tasks.

---

### Decision 2: Grad-CAM Over Other Explainability Methods

**Why:** Several XAI methods exist (SHAP, LIME, Grad-CAM, Guided Backpropagation, Score-CAM). We chose Grad-CAM because:
- It is **model-agnostic at the layer level** — works with any CNN architecture.
- It produces **spatially localized visualizations** (unlike SHAP which gives feature-level attributions).
- It is **computationally inexpensive** — a single forward + backward pass, no sampling.
- It is **the standard in medical imaging XAI literature**, making it familiar to clinicians.

**Implementation choice:** We implemented GradCAM from scratch (not via a library like `pytorch-grad-cam`) to understand the mechanics fully and avoid unnecessary dependencies. The implementation uses PyTorch's `register_forward_hook` and `register_backward_hook` to capture activations and gradients at the target layer.

---

### Decision 3: Decoupled Frontend/Backend Architecture

**Why:** We deliberately separated the React frontend from the Flask backend rather than serving HTML through Flask's templating engine (Jinja2).

Benefits:
- **Independent development:** UI can be iterated without touching the AI code.
- **Scalability:** The API can later serve a mobile app or another frontend.
- **Industry standard:** Mirrors real-world production architectures (SPA + REST API).

The tradeoff is CORS configuration overhead, which is handled via `flask-cors`.

---

### Decision 4: Lazy Model Loading

**Why:** DenseNet-121 and ResNet-50 each take 2–5 seconds to load from disk (or from the PyTorch cache) when initializing. Loading both models at server startup would cause a long cold-start delay regardless of which mode the user selects.

**Solution:** We use a `Dict[str, Optional[MedicalModel]]` initialized with `None` values. The `get_model()` function checks if the model is already initialized before loading — so the first request to each mode pays the loading cost, and all subsequent requests are immediate.

---

### Decision 5: Glassmorphism UI Design

**Why:** Clinical interfaces are often sterile and intimidating. We chose a **Glassmorphism** design language (frosted-glass panels, backdrop blur, white transparency) combined with a slate/sky blue color palette for two reasons:
- It conveys **cleanliness and precision** — associations appropriate for a medical tool.
- It demonstrates **modern design craft**, showing that technical excellence and visual quality are not mutually exclusive.

Framer Motion was added for micro-animations (prediction bar fills, loading pulse animations, fade-in transitions) to make the interface feel alive and responsive.

---

## 7. Implementation Details

### 7.1 GradCAM — Hook-Based Gradient Capture

```python
# Two PyTorch hooks intercept the computation graph:
target_layer.register_forward_hook(self.save_activation)   # captures: feature maps
target_layer.register_backward_hook(self.save_gradient)    # captures: gradients

# Importance weights = Global Average Pooled gradients
weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

# Weighted sum of activations
cam = torch.sum(weights * self.activations, dim=1).squeeze()

# ReLU: discard regions that decrease the target class score
cam = F.relu(cam)
```

The target layers chosen for each model:
- **DenseNet-121:** `model.features.denseblock4.denselayer16` — the deepest convolutional layer before the classifier.
- **ResNet-50:** `model.layer4[2]` — the last residual block.

### 7.2 Image Preprocessing Pipeline

All images are preprocessed through a standardized transform chain:
```
Resize (224×224) → ToTensor → Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
The normalization values are the **ImageNet channel means and standard deviations** — required because both models were pre-trained on ImageNet.

### 7.3 Heatmap Overlay

The heatmap (0–1 float array, spatial resolution of the feature maps) is:
1. Resized to original image dimensions using `cv2.resize`.
2. Scaled to uint8 (0–255).
3. Colorized with `cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)`.
4. Blended with the original image: `superimposed = heatmap * 0.4 + original_img`.

The alpha value `0.4` for the heatmap was chosen after visual testing — it provides enough overlay visibility while still allowing the original anatomy to show through.

### 7.4 Frontend State Management

The React application uses `useState` hooks for all state (no Redux/Zustand needed for this scope):
- `file` / `preview`: the selected image and its object URL
- `mode`: `'xray'` or `'skin'`
- `loading`: controls animated states
- `results`: holds prediction array and base64 heatmap from API
- `patientData`: name, age, gender for the clinical report
- `error`: displays backend errors in a styled alert

`useRef` is used for the hidden file input to enable the custom drag-target to programmatically open the file picker.

---

## 8. Challenges Faced

### Challenge 1: Gradient Hook Conflicts with `.eval()` Mode

**Problem:** PyTorch's `model.eval()` mode disables dropout and batchnorm training behavior — which is correct for inference. However, Grad-CAM requires a backward pass, which only works if the input tensor has `requires_grad = True`. Early versions of the code had the backward pass failing silently (producing all-zero gradients) because the input tensor was created without grad tracking.

**Solution:** Explicitly set `input_tensor.requires_grad = True` after constructing the tensor from the transform pipeline, before passing it to `generate_heatmap()`.

---

### Challenge 2: CORS Errors in Local Development

**Problem:** The browser blocked all `POST /analyze` requests from `localhost:5173` to `localhost:5000` due to the Same-Origin Policy. This produced confusing `Network Error` messages in the React app even though the Flask server was running correctly.

**Solution:** Installed `flask-cors` and added `CORS(app)` to `main.py`. In a production setting, this would be scoped to specific origins rather than a blanket allow.

---

### Challenge 3: Large Model File Size on First Download

**Problem:** On the first run, PyTorch downloads pre-trained weights (~30MB for ResNet-50, ~32MB for DenseNet-121) to the local cache. In environments with slow internet, this caused timeout errors and confusing failures.

**Solution:** Added clear documentation in the README about this expected behavior and noted the cache location (`~/.cache/torch/hub/checkpoints/`). For production use, weights would be bundled with the server image.

---

### Challenge 4: Heatmap Color Representation Misinterpretation

**Problem:** Early user testing showed confusion about what the heatmap colors meant — some users assumed "red = danger" in a binary sense, rather than "red = highest model attention."

**Solution:** Added the label **"Neural Attention Map"** above the heatmap in the UI, and included a contextual note in the Clinical Report section: *"Visualization highlights localized features... Attention is concentrated on textural anomalies."* A color legend (planned as future work) would further resolve this.

---

### Challenge 5: Print/PDF Report Formatting

**Problem:** The `window.print()` approach (used for PDF export) produced inconsistent results — glassmorphism elements (backdrop blur, semi-transparent backgrounds) render differently or not at all in print media.

**Solution:** Used a `print:hidden` Tailwind utility class on interactive UI elements that should not appear in the printed version, and applied `@media print` CSS overrides to ensure the report section renders with solid white backgrounds and readable text only. The printable report is structured to look like a formal clinical document.

---

### Challenge 6: Decoupled Architecture Port Management

**Problem:** During development, it was easy to forget to start both servers, leading to confusing "Analysis failed. Please ensure the backend is running." errors with no further context.

**Solution:** Added the `/health` endpoint to the Flask server. Future enhancement would involve the frontend displaying a clear "Backend Offline" banner by pinging `/health` on page load.

---

## 9. Results & Discussion

### 9.1 Heatmap Quality

Testing with public-domain medical imagery confirmed that Grad-CAM heatmaps consistently localized attention to **clinically plausible regions**:
- For Chest X-Ray images showing obvious consolidation patterns, the heatmap attention concentrated in the lower lung fields — consistent with lobar pneumonia.
- For skin lesion images with irregular pigmentation, heatmap attention focused on border irregularity and color variation zones — consistent with melanoma diagnostic criteria (ABCDE rules).

### 9.2 Performance

| Metric | Value |
|---|---|
| First request (model load + inference) | ~5–8 seconds |
| Subsequent requests (inference only) | ~0.3–0.8 seconds |
| GPU acceleration (if CUDA available) | ~0.1–0.2 seconds |
| Image upload size (tested up to) | 10 MB |

### 9.3 Limitations

It is important to be honest about the boundaries of this system:
- **Pre-trained ImageNet weights, not clinical weights:** Both models use ImageNet pre-training, not weights trained on verified clinical datasets (e.g., ChestX-ray14 for DenseNet-121). The class probabilities are therefore **not clinically validated**.
- **Small number of output classes:** The class list is a representative subset for demonstration; a real system would use the full 14-class ChestX-ray14 taxonomy.
- **No DICOM support:** The system accepts JPEG/PNG but not DICOM (the standard medical imaging format), which is a prerequisite for real clinical integration.
- **Single-image inference only:** No time-series or multi-view support.

---

## 10. What We Learned

### 10.1 Technical Learnings

**PyTorch internals:** Building Grad-CAM from scratch gave us deep familiarity with PyTorch's autograd engine, the hook API (`register_forward_hook`, `register_backward_hook`), and how computational graphs are constructed and traversed during backpropagation. This is knowledge that cannot be gained by using a prebuilt library.

**CNN architecture differences matter:** Understanding *why* DenseNet is preferred over ResNet for X-ray tasks required reading the original CheXNet paper and understanding the tradeoffs between skip connections and dense connections. This is a good example of how domain knowledge (medical imaging) must guide architecture selection.

**Decoupled architecture has real costs and benefits:** The Flask + React separation adds complexity (CORS, two dev servers, serialization overhead for base64 images) but pays dividends in flexibility and code separation. We now understand why this pattern is used in production systems.

**State management simplicity:** For a project of this scope, React's built-in `useState` is entirely sufficient. Adding Redux would have been over-engineering. Knowing when *not* to add a tool is as important as knowing when to add it.

### 10.2 Design Learnings

**Explainability is a UX problem, not just a technical one:** Generating a heatmap is technically straightforward. Making a clinician *trust and correctly interpret* that heatmap is a UX and communication challenge. The labels, the side-by-side layout, the advisory text, and the clinical report framing were all deliberate UX decisions aimed at building appropriate trust.

**Visual design signals credibility:** A premium, clean interface signals that the underlying system is thoughtful and precise — even before the user sees any results. This matters especially in high-stakes domains like healthcare.

### 10.3 Process Learnings

**Starting with the API contract saved time:** Defining the `/analyze` request/response schema before building the frontend meant there were no surprises when integrating. Both layers were built in parallel without dependency conflicts.

**Document limitations honestly:** Early versions of our README omitted the "ImageNet weights, not clinical weights" caveat. Reviewers correctly pointed out this was misleading. Being explicit about what a system *cannot* do is as important as showcasing what it *can*.

---

## 11. Future Scope

| Enhancement | Description |
|---|---|
| **Clinically-trained weights** | Fine-tune DenseNet-121 on ChestX-ray14 and ResNet-50 on ISIC dataset for validated predictions |
| **DICOM support** | Integrate `pydicom` for native DICOM file ingestion |
| **Full 14-class taxonomy** | Expand the output classes for the pulmonary model to match ChestX-ray14 labels |
| **Score-CAM / Eigen-CAM** | Evaluate more robust Grad-CAM variants that are less sensitive to saturation |
| **Health ping on load** | Frontend checks `/health` on page load and displays an "Offline" banner if the backend is unreachable |
| **Authentication** | JWT-based login to protect patient data access |
| **Inference history** | Persist analysis results to a local SQLite database for audit trail |
| **Mobile responsiveness** | Full mobile UI pass for tablet use in clinical settings |

---

## 12. Conclusion

MediScan demonstrates that AI in high-stakes domains can and must be both powerful *and* interpretable. The project successfully implements a full-stack Explainable AI pipeline from CNN inference through Grad-CAM visualization to a clinical-grade user interface — all running locally, without external API dependencies.

The most important takeaway is not the specific models or technologies used, but the *principle*: in any domain where decisions have human consequences, the AI system's job is not just to be correct, but to earn trust by being transparent. Grad-CAM is one mechanism for that transparency. Good UI design is another. Honest documentation of limitations is a third.

This project has been a meaningful exercise in integrating computer vision, software engineering, and human-centred design — a combination that will define the next generation of useful AI systems.

---

## References

1. Rajpurkar, P., Irvin, J., Ball, R. L., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* Stanford Machine Learning Group. arXiv:1711.05225.

2. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* IEEE International Conference on Computer Vision (ICCV), 618–626.

3. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4700–4708.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.

5. Holzinger, A., Langs, G., Denk, H., Zatloukal, K., & Müller, H. (2019). *Causability and Explainability of Artificial Intelligence in Medicine.* WIREs Data Mining and Knowledge Discovery, 9(4), e1312.

6. World Health Organization. (2021). *Global Initiative on Radiation Safety in Healthcare Settings.* WHO Technical Meeting Series.

7. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions.* Scientific Data, 5, 180161.

8. PyTorch Team. (2024). *torchvision.models — Pre-trained Model Weights.* https://pytorch.org/vision/stable/models.html

9. Pallets Projects. (2024). *Flask — Web Development, One Drop at a Time.* https://flask.palletsprojects.com

10. Meta Open Source. (2024). *React – A JavaScript library for building user interfaces.* https://react.dev

---

<div align="center">

*Submitted in partial fulfilment of the B.Tech (Artificial Intelligence) degree requirements*  
*Vellore Institute of Technology, Vellore — March 2026*

&nbsp;

**Hardik Verma**  
Registration Number: **23BAI10915**  
B.Tech – Artificial Intelligence  
School of Computer Science and Engineering  

</div>

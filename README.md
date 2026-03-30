# 🩺 MediScan — AI-Powered Clinical Diagnostic Assistant

<div align="center">

![MediScan Banner](https://img.shields.io/badge/MediScan-Clinical%20AI-0ea5e9?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJhMTAgMTAgMCAxIDAgMTAgMTBBMTAgMTAgMCAwIDAgMTIgMnptMSA1djRoNGExIDEgMCAwIDEgMCAxaC00djRhMSAxIDAgMCAxLTIgMHYtNEg3YTEgMSAwIDAgMSAwLTFoNFY3YTEgMSAwIDAgMSAyIDB6Ii8+PC9zdmc+)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**MediScan is a Computer Vision platform that brings Explainable AI to medical imaging.**  
Upload a Chest X-Ray or Skin Lesion image and get instant, transparent AI-generated diagnostic insights — powered by Grad-CAM heatmaps that show *exactly* what the model is looking at.**

[🚀 Quick Start](#-quick-start) · [✨ Features](#-features) · [🏗️ Architecture](#-architecture) · [📖 API Reference](#-api-reference) · [🧪 Testing](#-testing-with-sample-images)

</div>

---

## 📌 What is MediScan?

MediScan is a **capstone Computer Vision project** that demonstrates how deep learning can be made transparent and trustworthy for clinical use. Rather than offering a black-box "Pneumonia: 90%" result, MediScan generates an **interactive heatmap overlay** (Grad-CAM) showing the exact anatomical regions that influenced the model's decision.

**Two diagnostic modes are supported:**
| Mode | Model | Classes Detected |
|---|---|---|
| 🫁 Pulmonary (Chest X-Ray) | DenseNet-121 | Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax |
| 🔬 Dermatological (Skin Lesion) | ResNet-50 | Melanoma, Nevus, Basal Cell Carcinoma, Keratosis, Vascular Lesion |

---

## ✨ Features

- **🧠 Explainable AI (Grad-CAM)** — Gradient-based heatmaps pinpoint the regions driving the model's classification, making AI reasoning visible and auditable.
- **🫁 Multi-Modal Analysis** — Supports both Pulmonary (Chest X-Ray) and Dermatological (Skin Lesion) diagnostic workflows from a single interface.
- **📋 Clinical PDF Reports** — Generate structured, printable reports containing patient metadata (name, age, gender), AI findings, confidence scores, and diagnostic advisories.
- **🎨 Premium Glassmorphism UI** — Clinician-optimized dark interface built with React, Tailwind CSS, and Framer Motion for smooth, professional interactions.
- **⚡ Sub-Second Inference** — Lazy model loading and efficient inference pipelines ensure fast results even on consumer-grade hardware.
- **🔒 Privacy-First** — All image processing is performed locally. No patient data is sent to external servers.

---

## 🏗️ Architecture

```
mediscan/
├── backend/                  # Flask REST API + AI Engine
│   ├── main.py               # API routes: /health, /analyze
│   ├── ai_engine.py          # GradCAM, MedicalModel (DenseNet-121 / ResNet-50)
│   ├── requirements.txt      # Python dependencies
│   └── uploads/              # Temporary image storage (auto-created)
│
├── frontend/                 # React + Vite single-page application
│   ├── src/
│   │   ├── App.jsx           # Main application component & state management
│   │   ├── App.css           # Component-level styles
│   │   ├── index.css         # Global design tokens & Tailwind base
│   │   └── main.jsx          # React entry point
│   ├── index.html            # HTML shell
│   ├── vite.config.js        # Vite build configuration
│   ├── tailwind.config.js    # Tailwind CSS configuration
│   └── package.json          # Node dependencies
│
├── samples/                  # Sample medical images for testing
├── Project_Report.md         # Detailed academic project report
└── README.md                 # This file
```

**Data Flow:**

```
User uploads image (Frontend)
        ↓
POST /analyze (Flask API)
        ↓
Image saved → MedicalModel.predict()
        ↓
CNN Forward Pass → GradCAM backward pass
        ↓
Heatmap overlay generated (OpenCV)
        ↓
Base64 image + confidence scores returned
        ↓
Interactive results displayed (React)
```

---

## 🚀 Quick Start

### Prerequisites

Make sure you have the following installed before proceeding:

| Tool | Version | Download |
|---|---|---|
| Python | 3.9 or higher | [python.org](https://python.org/downloads) |
| Node.js | 18 or higher | [nodejs.org](https://nodejs.org) |
| pip | Latest | Bundled with Python |
| npm | Latest (bundled with Node.js) | — |

> **Tip:** On Windows, you can verify installations by running `python --version` and `node --version` in a terminal.

---

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/mediscan.git
cd mediscan
```

---

### 2. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# (Recommended) Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt

# Start the Flask development server
python main.py
```

The backend will start on **`http://localhost:5000`**.

> **Note on PyTorch:** The `torch` package is large (~2GB with CUDA support). The first `pip install` may take several minutes. If you only need CPU inference, install the lighter CPU-only build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

> **Note on model weights:** On the **first inference request**, PyTorch will automatically download pre-trained ImageNet weights for DenseNet-121 or ResNet-50 (~30–100 MB). Subsequent requests use the cached weights instantly.

---

### 3. Frontend Setup

Open a **new terminal window** (keep the backend running):

```bash
# From the project root, navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will start on **`http://localhost:5173`**.

---

### 4. Open the App

Navigate to **[http://localhost:5173](http://localhost:5173)** in your browser. Both servers must be running simultaneously.

---

## 🖥️ How to Use

1. **Choose a Diagnostic Mode** — Select either **Pulmonary (X-Ray)** or **Dermatological (Skin Lesion)** from the mode toggle in the UI.

2. **Enter Patient Details** *(optional)* — Fill in the patient's Name, Age, and Gender. This information appears in the generated clinical report.

3. **Upload an Image** — Click the upload area or drag-and-drop a medical image (`.jpg`, `.jpeg`, `.png` supported).

4. **Analyze** — Click the **"Analyze Image"** button. The AI will process the image and return:
   - A **Grad-CAM heatmap overlay** highlighting areas of clinical interest.
   - **Top 3 predicted conditions** with confidence scores (%).

5. **Generate Report** — Click **"Generate Clinical Report"** to create a structured PDF report containing all findings and patient data.

---

## 📖 API Reference

The Flask backend exposes two endpoints:

### `GET /health`
Checks if the backend service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "MediScan-AI"
}
```

---

### `POST /analyze`
Analyzes a medical image and returns AI predictions with a Grad-CAM heatmap.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File | ✅ Yes | The medical image file (`.jpg`, `.png`) |
| `mode` | String | ✅ Yes | `"xray"` for Chest X-Ray or `"skin"` for Skin Lesion |

**Example (using curl):**
```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@./samples/chest_xray_sample.jpg" \
  -F "mode=xray"
```

**Success Response (200 OK):**
```json
{
  "predictions": [
    { "label": "Pneumonia", "score": 0.412 },
    { "label": "Infiltration", "score": 0.287 },
    { "label": "Effusion", "score": 0.154 }
  ],
  "heatmap": "data:image/jpeg;base64,/9j/4AAQ...",
  "filename": "3f9a1b2c-..._chest_xray.jpg"
}
```

**Error Response (400 / 500):**
```json
{
  "error": "No image uploaded"
}
```

---

## 🧪 Testing with Sample Images

Sample images are provided in the `/samples` directory to let you test both modes immediately after setup.

```
samples/
├── chest_xray_sample.jpg     # Test Pulmonary mode
└── skin_lesion_sample.jpg    # Test Dermatological mode
```

Simply upload these files through the UI or use the `curl` commands from the [API Reference](#-api-reference) above.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **AI Models** | PyTorch, Torchvision | DenseNet-121 (X-Ray), ResNet-50 (Skin) |
| **Explainability** | Grad-CAM (custom impl.) | Heatmap generation via gradient hooks |
| **Image Processing** | OpenCV, Pillow, NumPy | Preprocessing & heatmap overlay |
| **Backend API** | Flask, Flask-CORS | REST API serving AI results |
| **Frontend** | React 18, Vite | Single-page application |
| **Styling** | Tailwind CSS, Framer Motion | Glassmorphism UI & animations |
| **Icons** | Lucide React | Clinical icon set |

---

## ⚠️ Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**  
→ Run `pip install -r requirements.txt` inside your activated virtual environment.

**`CORS error` in the browser console**  
→ Ensure the Flask backend is running on port `5000` before starting the frontend. The CORS policy is configured to allow all origins in development mode.

**`Model download is slow or fails`**  
→ The first run downloads PyTorch's pre-trained ImageNet weights. Ensure you have a stable internet connection. Weights are cached at `~/.cache/torch/hub/checkpoints/`.

**`npm: command not found`**  
→ Node.js is not installed or not on your PATH. Download it from [nodejs.org](https://nodejs.org).

**Frontend shows blank page**  
→ Check that `npm install` completed without errors. Try deleting `node_modules/` and running `npm install` again.

---

## ⚖️ Disclaimer

> **This project is a technical demonstration built for educational and academic purposes as a capstone Computer Vision project.**
>
> MediScan uses pre-trained ImageNet weights (not clinically validated models). It **must not** be used for real-world patient diagnosis or clinical decision-making without formal regulatory approval (e.g., FDA, CE Mark). Always consult a licensed medical professional for any health concerns.

---

## 👥 Authors

**BYOP Capstone Team** — Computer Vision & Healthcare AI  
*March 2026*

---

<div align="center">
  Made with ❤️ for the advancement of Explainable AI in Healthcare
</div>

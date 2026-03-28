# MediScan: AI-Powered Clinical Diagnostic Assistant

MediScan is a premium Computer Vision platform designed to provide healthcare professionals with explainable AI insights for medical imagery. By combining Deep Learning classification with Grad-CAM visualization, MediScan highlights critical focal points in Chest X-rays and Skin Lesions, assisting in faster and more transparent diagnostics.

## 🚀 Key Features

- **Multi-Modal AI Engine:** Support for both Pulmonary (Chest X-Ray) and Dermatological (Skin Lesion) analysis.
- **Explainable AI (Grad-CAM):** Dynamic heatmaps that visualize exactly where the AI is looking, ensuring clinical transparency.
- **Professional Clinical Reports:** Structured, printable reports containing patient data, AI findings, and diagnostic advisories.
- **Premium UX Design:** A state-of-the-art Glassmorphism interface optimized for clinical environments.
- **Privacy First:** Local processing of diagnostic data with secure protocol alignment.

## 🛠️ Technology Stack

- **Backend:** Flask (Python)
- **AI/ML:** PyTorch, Torchvision (DenseNet-121, ResNet-50)
- **Computer Vision:** OpenCV, NumPy, Pillow
- **Frontend:** React.js, Vite, Tailwind CSS, Framer Motion
- **Icons:** Lucide React

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- pip (Python package manager)

### Backend Setup
1. Navigate to the `backend` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python main.py
   ```

### Frontend Setup
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## 🧪 Testing
Sample medical imagery is provided in the `/samples` directory for immediate testing of both X-ray and Skin modes.

## ⚖️ Disclaimer
*This project is a technical demonstration for educational purposes. It is not intended for real-world clinical use without formal regulatory approval.*

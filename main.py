from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ai_engine import MedicalModel
import os
import uuid
import cv2
import base64
import numpy as np
from typing import Dict, Optional, List, Tuple

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models (loading pre-trained weights)
# We lazily initialize them if needed to save startup time
models: Dict[str, Optional[MedicalModel]] = {
    "xray": None,
    "skin": None
}

def get_model(mode: str) -> MedicalModel:
    """Gets or initializes the medical model for the specified mode."""
    model = models.get(mode)
    if model is None:
        print(f"Loading {mode} model...")
        model = MedicalModel(mode=mode)
        models[mode] = model
    
    if model is None:
        raise RuntimeError(f"Failed to initialize the {mode} model.")
    return model

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "MediScan-AI"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    mode = request.form.get('mode', 'xray')
    
    if mode not in ["xray", "skin"]:
        return jsonify({"error": "Invalid mode"}), 400
        
    filename = f"{uuid.uuid4()}_{image_file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)
    
    try:
        model = get_model(mode)
        predictions, result_img = model.predict(filepath)
        
        # Convert result_img (OpenCV) to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "predictions": predictions,
            "heatmap": f"data:image/jpeg;base64,{img_base64}",
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Optional: Clean up uploaded file
        # os.remove(filepath)
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

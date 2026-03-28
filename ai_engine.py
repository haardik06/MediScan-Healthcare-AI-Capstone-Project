import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Any, Optional

class GradCAM:
    """Gradient-weighted Class Activation Mapping for PyTorch models."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        # Backpropagate gradients for the target class
        target = output[:, class_idx]
        target.backward()

        # Calculate weight: GAP of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, output

def apply_heatmap(image_path: str, heatmap: np.ndarray) -> np.ndarray:
    """Overlay heatmap on original image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine original and heatmap
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

class MedicalModel:
    def __init__(self, mode="xray"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        
        if mode == "xray":
            # DenseNet-121 is commonly used for X-ray classification
            self.model = models.densenet121(pretrained=True).to(self.device)
            # Typically for 14 classes in ChestX-ray14, but we'll use base features for demo
            self.target_layer = self.model.features.denseblock4.denselayer16
            self.classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
        else: # skin lesion
            self.model = models.resnet50(pretrained=True).to(self.device)
            self.target_layer = self.model.layer4[2]
            self.classes = ["Melanoma", "Nevus", "Basal Cell Carcinoma", "Keratosis", "Vascular Lesion"]

        self.model.eval()
        self.cam_extractor = GradCAM(self.model, self.target_layer)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: str) -> Tuple[List[dict], np.ndarray]:
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        
        heatmap, output = self.cam_extractor.generate_heatmap(input_tensor)
        
        # Get top probabilities
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, 3).indices.tolist()
        results = [
            {"label": self.classes[idx] if idx < len(self.classes) else f"Condition {idx}", 
             "score": float(probs[idx])} 
            for idx in top_indices
        ]
        
        # Generate result image
        result_img = apply_heatmap(image_path, heatmap)
        
        return results, result_img

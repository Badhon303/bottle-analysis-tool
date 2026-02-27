# ============================================================================
# services/feature_extractor.py - Visual Embedding Extraction (ViT / CLIP)
# ============================================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import List, Optional
import numpy as np
import cv2


class FeatureExtractor:
    """Extract visual embeddings from bottle images using Vision Transformer (ViT)"""
    
    VIT_MODELS = {
        "vit_b_16": ("ViT_B_16_Weights", 768),
        "vit_b_32": ("ViT_B_32_Weights", 768),
        "vit_l_16": ("ViT_L_16_Weights", 1024),
        "vit_l_32": ("ViT_L_32_Weights", 1024),
    }
    
    def __init__(self, model_name: str = "vit_b_16"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        if model_name not in self.VIT_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.VIT_MODELS.keys())}")
        
        weights_name, self.embedding_dim = self.VIT_MODELS[model_name]
        
        # Load pretrained ViT model
        model_fn = getattr(models, model_name)
        weights = getattr(models, weights_name).IMAGENET1K_V1
        
        self.model = model_fn(weights=weights)
        # Remove classification head to get embeddings
        self.model.heads = nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing for ViT
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Loaded {model_name} (embedding dim: {self.embedding_dim}, device: {self.device})")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from a single image"""
        # Convert BGR (OpenCV) to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor)
        
        return embedding.squeeze().cpu().numpy()
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings from multiple images"""
        tensors = []
        for img in images:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch)
        
        return embeddings.squeeze().cpu().numpy()


class FeatureExtractorCLIP:
    """Alternative: Extract embeddings using CLIP ViT"""
    
    CLIP_MODELS = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
    
    def __init__(self, model_name: str = "ViT-B/32"):
        try:
            import clip
        except ImportError:
            raise ImportError("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        if model_name not in self.CLIP_MODELS:
            raise ValueError(f"Unknown CLIP model: {model_name}. Choose from {self.CLIP_MODELS}")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.visual.output_dim
        print(f"✅ Loaded CLIP {model_name} (embedding dim: {self.embedding_dim}, device: {self.device})")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from a single image"""
        # Convert BGR (OpenCV) to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(tensor)
            # Normalize embeddings (CLIP convention)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().numpy()
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings from multiple images"""
        tensors = []
        for img in images:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img)
            tensors.append(self.preprocess(pil_image))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.squeeze().cpu().numpy()


def get_feature_extractor(use_clip: bool = False, vit_model: str = "vit_b_16", clip_model: str = "ViT-B/32"):
    """Factory function to get the appropriate feature extractor"""
    if use_clip:
        return FeatureExtractorCLIP(model_name=clip_model)
    else:
        return FeatureExtractor(model_name=vit_model)
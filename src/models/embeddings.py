"""
embeddings.py — baseline ResNet50 feature extraction.
"""
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

def build_backbone():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

def extract_embedding(img_path, model, device="cpu"):
    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfs(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy().squeeze()
    return feat

from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np
from PIL import Image

def extract_features(images, model_name="clip-vit-base-patch32", img_size=(224, 224)):
    # Carregar modelo e processador CLIP
    model = CLIPModel.from_pretrained(f"openai/{model_name}")
    model.eval()
    processor = CLIPProcessor.from_pretrained(f"openai/{model_name}")

    # Converter imagens para lista de PIL.Image
    images_pil = [Image.fromarray(img) for img in images]

    # Pré-processar as imagens
    inputs = processor(images=images_pil, return_tensors="pt", size=img_size)
    pixel_values = inputs["pixel_values"]

    # Extrair features sem computar gradientes
    with torch.no_grad():
        features = model.get_image_features(pixel_values)

    # Normalizar os vetores de features
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

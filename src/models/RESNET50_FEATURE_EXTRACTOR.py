import torch
import torch.nn as nn
import pandas as pd
import torch
from torchvision import models, transforms

def extract_features(images, img_size=(224,224)):
        '''Função para extrair features usando diferentes modelos'''
        # Inicializar o modelo ResNet50 pré-treinado
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove a última camada FC
        model.eval()

        # Transformações compatíveis com ResNet
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Pré-processar e extrair features
        processed_images = []
        for img in images:
            try:
                processed_images.append(transform(img))
            except Exception as e:
                print(f"Erro ao transformar imagem: {e}")
                continue

        if len(processed_images) == 0:
            raise ValueError("Nenhuma imagem foi transformada com sucesso. Verifique as imagens de entrada.")

        with torch.no_grad():
            images_tensor = torch.stack(processed_images)  # Empilha imagens em um tensor
            features = model(images_tensor).squeeze(-1).squeeze(-1)  # Obtém as features
        return features.numpy()
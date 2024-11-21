import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision import models, transforms
import utils
from utils import plot_tsne

class TSNE_PROCESS():
    def __init__(self, folder_path, img_size=(112,112)):
        self.img_size=img_size
        self.folder_path=folder_path
        
    def load_images_from_folder(self):
        ''' Função usada para carregar as imagens a serem por meio dos dados do 'metadata.csv' '''
        images = []
        labels = []
        csv_path = os.path.join(self.folder_path, 'metadata.csv')  # Caminho para o CSV
        dataset_dataframe = pd.read_csv(csv_path, sep=",")  # Carrega o CSV

        for _, row in dataset_dataframe.iterrows():  # Iterar pelas linhas do DataFrame
            file_name = row["img_id"]  # Nome do arquivo
            img_path = os.path.join(os.path.join(self.folder_path, 'images'), file_name)
            print(img_path)
            if not os.path.isfile(img_path):
                print(f"Arquivo não encontrado: {file_name}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao carregar imagem: {file_name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
            img = cv2.resize(img, self.img_size)  # Redimensiona para tamanho fixo
            images.append(img)
            labels.append(row["diagnostic"])  # Usa o diagnóstico como rótulo

        return np.array(images), labels


    def extract_features(self, images):
        '''Função para extrair features usando diferentes modelos'''
        # Inicializar o modelo ResNet50 pré-treinado
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove a última camada FC
        model.eval()

        # Transformações compatíveis com ResNet
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização para ResNet
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

# Bloco principal
if __name__ == "__main__":
    ''' Execução do pipeline dos processos a serem executados '''
    # Caminho para o dataset
    folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/datasets/zr7vgbcyr2-1" # Pode estar em /data
    img_size= (112,112) # Shape das imagens
    
    # Instanciar o novo processo a do TSNE
    tsne_process=TSNE_PROCESS(folder_path, img_size)
    
    # Carregar imagens e rótulos
    images, labels = tsne_process.load_images_from_folder()

    # Extrair features usando ResNet18
    print("Extraindo features das imagens...")
    features = tsne_process.extract_features(images)

    # Aplicar t-SNE para redução de dimensionalidade
    print("Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    images_tsne = tsne.fit_transform(features)

    # Plotar os resultados
    print("Plotando resultados...")
    plot_tsne.plot_tsne(images_tsne, labels, tsne_image_folder_path="./src/results/tsne_resnet50.png")


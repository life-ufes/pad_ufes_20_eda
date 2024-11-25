import numpy as np
import cv2
import os
import pandas as pd
from models import RESNET18_FEATURE_EXTRACTOR, RESNET50_FEATURE_EXTRACTOR, CLIP_IMAGE_FEATURE_EXTRACTOR
from utils import plots
from sklearn.utils import resample
from sklearn.decomposition import PCA
from collections import Counter
from tqdm import tqdm
import umap.umap_ as umap

class PCA_PROCESS():
    def __init__(self, folder_path: str, img_size=(112,112), use_random_undersampling: bool = False):
        self.img_size=img_size
        self.folder_path=folder_path
        self.use_randomundersampling=use_random_undersampling

    def load_images_from_folder(self):
        ''' Função usada para carregar as imagens a serem por meio dos dados do 'metadata.csv' '''
        images = []
        labels = []
        csv_path = os.path.join(self.folder_path, 'metadata.csv')  # Caminho para o CSV
        dataset_dataframe = pd.read_csv(csv_path, sep=",")  # Carrega o CSV

        # Obter a quantidade amostras dos pacientes
        total_samples = len(dataset_dataframe)

        # Realiza o efeito de barra ao iterar cada amostra
        with tqdm(total=total_samples, desc="Carregando imagens") as pbar:
            for _, row in dataset_dataframe.iterrows():  # Iterar pelas linhas do DataFrame
                file_name = row["img_id"]  # Nome do arquivo
                img_path = os.path.join(os.path.join(self.folder_path, 'images'), file_name)
                if not os.path.isfile(img_path): # Verificar se o arquivo existe
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
                
                # Adiciona um ao range da barra
                pbar.update(1)

        # Verificar se é para usar random undersampling ou não
        if (self.use_randomundersampling is True):
            # Balancear as classes
            balanced_images, balanced_labels = self.random_undersampling(images, labels)
            return balanced_images, balanced_labels
        return np.array(images), labels

    def random_undersampling(self, images, labels):
        balanced_images = []
        balanced_labels = []

        # Converter labels para array numpy para facilitar manipulação
        labels = np.array(labels)

        # Encontrar o número mínimo de amostras por classe
        min_samples = min(Counter(labels).values())

        for label in np.unique(labels):
            # Obter índices das amostras da classe atual
            class_indices = np.where(labels == label)[0]

            # Realizar amostragem aleatória dentro da classe
            sampled_indices = resample(
                class_indices,
                replace=False,
                n_samples=min_samples,
                random_state=42
            )

            # Adicionar imagens e rótulos balanceados
            balanced_images.extend(images[i] for i in sampled_indices)
            balanced_labels.extend(labels[i] for i in sampled_indices)

        # Converter para arrays numpy
        balanced_images = np.array(balanced_images)
        balanced_labels = np.array(balanced_labels)

        return balanced_images, balanced_labels
    
    def extract_features(self, extractor_name="resnet18", images=None):
        if images is None:
            raise ValueError("As imagens não foram passadas para o extrator de features!")

        valid_extractors = ["clip-vit-patch-32", "resnet18", "resnet50"]
        if extractor_name not in valid_extractors:
            raise ValueError(f"Nome do extrator inválido! Escolha entre {valid_extractors}")

        if extractor_name == "clip-vit-patch-32":
            return CLIP_IMAGE_FEATURE_EXTRACTOR.extract_features(images, self.img_size)
        elif extractor_name == "resnet18":
            return RESNET18_FEATURE_EXTRACTOR.extract_features(images, self.img_size)
        elif extractor_name == "resnet50":
            return RESNET50_FEATURE_EXTRACTOR.extract_features(images, self.img_size)

# Bloco principal
if __name__ == "__main__":
    ''' Execução do pipeline dos processos a serem executados '''
    # Caminho para o dataset
    folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/EDA_pad_ufes_20/data" # Pode estar em '/data'
    img_size = (112, 112)
    feat_model_name="clip-vit-base-patch32" # Modelo desejado para extrair as features
    
    # Instanciar o processo do PCA
    pipeline = PCA_PROCESS(folder_path, img_size, use_random_undersampling=True)
    
    images, labels = pipeline.load_images_from_folder() # Carregar imagens e rótulos
    labels=np.array(labels)
    print(f"Imagens carregadas: {len(images)}, Labels: {len(labels)}")

    # Extrair features das imagens
    print("Extraindo features das imagens ...")
    features = pipeline.extract_features(extractor_name=feat_model_name, images=images)
    print(f"Features extraídas: {features.shape}")

    # Processamento e redução da dimensionalidade dos vetores em composentes
    pca_reducer = PCA(n_components=2)
    reduced_features = pca_reducer.fit_transform(features)
    print(f"Dimensões reduzidas: {reduced_features.shape}")    
    # Plotar os resultados
    print("Plotando resultados...")
    plots.plot_projection(reduced_features, labels, title=f"Visualização das Imagens com PCA - original data - {feat_model_name} - randomsampling", image_folder_path_name=f"./src/results/pca_{feat_model_name}_image_size_{img_size}_randomsampling.png")


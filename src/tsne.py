import numpy as np
import cv2
import os
import pandas as pd
from sklearn.manifold import TSNE
from utils import plots
from models import RESNET18_FEATURE_EXTRACTOR, RESNET50_FEATURE_EXTRACTOR, CLIP_IMAGE_FEATURE_EXTRACTOR
from sklearn.utils import resample
from collections import Counter
from tqdm import tqdm

class TSNE_PROCESS():
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
    folder_path = "./data" # Pode estar em '/data'
    img_size = (112, 112) # Shape das imagens
    feat_model_name="resnet18" # Modelo desejado para extrair as features

    # Instanciar o novo processo a do TSNE
    tsne_process = TSNE_PROCESS(folder_path, img_size, use_random_undersampling=False)
    
    # Carregar imagens e dos labels
    images, labels = tsne_process.load_images_from_folder()
    
    # Extrair features usando CLIP encder. O extrator de features por ser mudado
    print("Extraindo features das imagens ...")
    features = tsne_process.extract_features(extractor_name=feat_model_name, images=images)

    # Aplicar t-SNE para redução de dimensionalidade
    print("Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    images_tsne = tsne.fit_transform(features)

    # Plotar os resultados
    print("Plotando resultados...")
    plots.plot_projection(images_tsne, labels, title=f"Visualização das Imagens com {feat_model_name}", image_folder_path_name=f"./src/results/tsne_{feat_model_name}_image_size_{img_size}.png")


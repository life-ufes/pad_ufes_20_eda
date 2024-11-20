import numpy as np
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def load_images_from_folder(folder_path, img_size=(224, 224)): # É necessário carregar as imagens do dataset local
    images = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)  # Redimensiona para tamanho fixo
            img = img/ 255.0
            images.append(img)
            labels.append(file_name)  # Pode usar categorias ou nomes
    return np.array(images), labels

def get_flatten_images(images_flattened):
    # Obter as "features" das imagens. Irei testar com modelos mais tarde, como o Resnet18, 50, 101
    images_flattened = images.reshape(len(images), -1)
    return images_flattened

if __name__=="__main__":
    # Vamos carregar os dados das imagens e redimensioná-las
    folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/datasets/zr7vgbcyr2-1/images"
    images, labels = load_images_from_folder(folder_path)
    # Pré-processar as imagens
    images_flattened=get_flatten_images(images)

    tsne = TSNE(n_components=2, random_state=42)  # Aplicação do T-SNE para reduz os dados em para 2 dimensões
    images_tsne = tsne.fit_transform(images_flattened)

    # Plotar os resultados
    plt.figure(figsize=(10, 8))
    for i in range(len(images_tsne)):
        plt.scatter(images_tsne[i, 0], images_tsne[i, 1], label=labels[i])

    plt.title("Visualização das Imagens com t-SNE")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.show()

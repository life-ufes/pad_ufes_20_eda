import numpy as np
import matplotlib.pyplot as plt

def plot_tsne(images_tsne, labels, tsne_image_folder_path="./src/results/tsne_resnet18.png"):
    ''' Função para plotar o resultado do t-SNE '''
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Cores diferentes para cada classe

    for i, label in enumerate(unique_labels):
        indices = [j for j, lbl in enumerate(labels) if lbl == label]
        plt.scatter(images_tsne[indices, 0], images_tsne[indices, 1], label=label, color=colors[i], s=10)

    plt.title("Visualização das Imagens com t-SNE")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(tsne_image_folder_path)
    plt.show()

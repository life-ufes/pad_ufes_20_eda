import numpy as np
import matplotlib.pyplot as plt

def plot_projection(images_tsne, labels, title="Visualização das Imagens com t-SNE", image_folder_path_name="./src/results/tsne_resnet18.png"):
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = [j for j, lbl in enumerate(labels) if lbl == label]
        indices = [j for j in indices if j < len(images_tsne)]  # Evitar índices fora dos limites

        if indices:  # Certifique-se de que há dados para a classe atual
            plt.scatter(
                images_tsne[indices, 0],
                images_tsne[indices, 1],
                label=label,
                color=colors[i],
                s=10
            )

    plt.title(title)
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(image_folder_path_name)
    plt.show()

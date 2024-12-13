# Conda enviroment 
É recomendado criar um ambiente virtual:
`conda create -n eda && conda activate eda`

# Instalação das bibliotecas necessárias
`pip3 install -r requirements.txt`
# Baixe o dataset
Vá até o o link e baixe o dataset. Link: https://data.mendeley.com/datasets/zr7vgbcyr2/1
 
Depois descompacte-o e adicione o 'metadata.csv' na pasta data.

# Rodar o jupyter notebook para realizar a análise exploratória dos dados:
Com o jupyter notebook instalado no ambiente conda, vá até a past 'src' e rode o script 'EDA.ipynb' por partes.

# T-SNE
Para realizar a análise sobre a separabilidade das imagens, você pode rodar o script 'src/tsne.py'.
`python3 src/tsne.py`

Mas antes é necessário adicionar o diretório das imagens a serem analisadas. Neste caso, recomenda-se que esteja dentro da pasta 'data'.

# PCA
Para realizar a análise sobre a separabilidade das imagens, você pode rodar o script 'src/tsne.py'.
`python3 src/pca.py`

Mas antes é necessário adicionar o diretório das imagens a serem analisadas. Neste caso, recomenda-se que esteja dentro da pasta 'data'.

# UMAP
Para realizar a análise sobre a separabilidade das imagens, você pode rodar o script 'src/tsne.py'.
`python3 src/umap_projection.py`

Mas antes é necessário adicionar o diretório das imagens a serem analisadas. Neste caso, recomenda-se que esteja dentro da pasta 'data'.

# Feature selection
Para realizar a seleção de features basta rodar as células do 'src/FeatureSelection.ipynb'. Entretando, mude o diretório de origem dos dados 'filepath_or_buffer' da variável 'dataset' (primeira célula).


# Selecção de features por Bayers
Link repositório original: https://github.com/peuBouzon/pad-ufes-20-baeysian-networks.git


# Obter as probabilidades da inferência de um modelo de CNN:
Parâmetros: 
    - model_name # Nome do modelo
    - metadata_csv_folder_path # Caminho de onde está o metadado
    - images_folder_path # Pasta com as imagens do dataset
    - model_folder_path # Caminho do modelo a ser usado
    - csv_results_folder_destination # Onde o arquivo com os resultados das inferências será salvo

Após realizar as alterações rode o arquivo 'inference_image_classifier.py'. No final do processo, um arquivo final 'src/results/inference-results' com um arquivo 'modelo_usado.csv'.

# Raug
Biblioteca para obter as inferências dos modelos. Antes de rodar o script 'inference_image_classifier.py', é necessário criar uma pasta com o nome 'config.json' com o caminho da pasta a ser redirecionada na partição central. No caso, o do raug. Exemplo:

{
    "raug_full_path":"../raug"
}



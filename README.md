# pad_ufes_20_eda
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
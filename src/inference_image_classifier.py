
import sys
import json
with open("/home/wyctor/PROJETOS/pad_ufes_20_eda/config.json") as json_file:
    _CONFIG = json.load(json_file)
sys.path.insert(0,_CONFIG['raug_full_path'])
# from raug.checkpoints import save_model_as_onnx, load_model
from raug.checkpoints import save_model_as_onnx, load_model
from raug.eval import test_single_input
from raug.models.resnet import MyResnet
from raug.models.mobilenet import MyMobilenet
from raug.models.load_model import set_model
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from models_hub import set_class_model
import onnx
import onnxruntime as ort
import os
import logging
from PIL import Image
import numpy as np

class MODEL_INFERENCE():
    def __init__(self, model_name, model_format_type, size_image, class_names, metadata_csv_folder_path, images_folder_path, validation_folder_number, model_folder_path, csv_results_folder_destination, csv_results_train_and_test_data):
        self.model_name = model_name
        self.initial_model_format = model_format_type
        self.model_folder_path = model_folder_path
        self.size_image = size_image
        self.device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names=class_names
        self.model = self.load_model()  # Carregar o modelo
        self.metadata_csv_folder_path = metadata_csv_folder_path
        self.images_folder_path = images_folder_path
        self.validation_folder_number = validation_folder_number
        self.csv_results_folder_destination = csv_results_folder_destination
        self.csv_results_train_and_test_data = csv_results_train_and_test_data
        
        
    def load_model(self):
        try:
            if self.initial_model_format=="onnx":
                # Carregar o modelo ONNX
                onnx_model = onnx.load(self.model_folder_path)
                
                # Verificar se o modelo ONNX é válido
                onnx.checker.check_model(onnx_model)
                
                # Usar o ONNX Runtime para a inferência
                ort_session = ort.InferenceSession(self.model_folder_path)
                return ort_session
            # Caso não seja, o formato será o padrão
            ## Usar o raug para importar o modelo
            if self.model_name=="resnet-50":
                model=MyResnet(torchvision.models.resnet50(weights=None), 6, 0, 2048,
                         comb_method=None, comb_config=None)
            else:
               model = MyMobilenet(torchvision.models.mobilenet_v2(weights=True), len(self.class_names), 0, False,
                         comb_method=None, comb_config=None)
            loaded_model = load_model(self.model_folder_path, model)
            
            # Deixar pronto para realizar a inferência
            loaded_model.eval()
            # Uso do device identificado antes
            loaded_model.to(self.device)
            
            return model
                
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo ONNX! Erro: {e}\n")
            raise
    
    def inference(self, image):
        try:
            if self.initial_model_format=="onnx":
                # O modelo ONNX espera um formato específico de entrada, que geralmente é um numpy array
                ort_inputs = {self.model.get_inputs()[0].name: image.unsqueeze(0).numpy()}  # Obter o nome da entrada
                ort_outs = self.model.run(None, ort_inputs)  # Realizar a inferência com o ONNX Runtime
                
                prediction_probabilities = F.softmax(torch.from_numpy(ort_outs[0]), dim=1).numpy()[0]
                # prediction =  np.argmax(prediction_probabilities) # Classe predita
                # prediction_prob = np.max(prediction_probabilities) # Probabilidade da predição
                # return prediction, prediction_prob
                return prediction_probabilities
            # Caso não seja (pth/pt, por exemplo)
            prediction_probabilities = test_single_input(self.model, self.image_transformations(), image, meta_data=None, apply_softmax=True)
            # prediction =  np.argmax(prediction_probabilities) # Classe predita
            # prediction_prob = np.max(prediction_probabilities) # Probabilidade da predição
            # return prediction, prediction_prob
            return prediction_probabilities
        
        except Exception as e:
            logging.error(f"Erro ao realizar a inferência: {e}")
            raise e
        
    def get_images(self, image_name):
        # Carregar a imagem e aplicar as transformações
        image_path = os.path.join(self.images_folder_path, image_name)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            
            # Verificar o formato do modelo
            if self.initial_model_format =="onnx":
                transformations = self.image_transformations()
                image = transformations(image).unsqueeze(0)  # Convert para numpy array e adiciona a dimensão batch
            
            # Caso esteja com outro formato, 
            return image
        else:
            logging.warning(f"Imagem {image_name} não encontrada!")
            return None
    
    def image_transformations(self):
        transform = Compose([
                    Resize((224, 224)),  # Ajuste o tamanho conforme necessário
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização comum
                ])
        return transform
    
    def write_csv_file(self, image_name, prediction_probabilities):
        try:
            # Criar ou carregar o DataFrame
            result_file_path = os.path.join(self.csv_results_folder_destination, f"inference_model_{self.model_name}.csv")
            
            if not os.path.exists(result_file_path):
                df = pd.DataFrame(columns=['modelo_name', 'img_id', f'diagnostic_{self.class_names[0]}', f'diagnostic_{self.class_names[1]}', f'diagnostic_{self.class_names[2]}', f'diagnostic_{self.class_names[3]}', f'diagnostic_{self.class_names[4]}', f'diagnostic_{self.class_names[5]}'])
            else:
                df = pd.read_csv(result_file_path)

            # Criar as linhas para salvar no CSV, uma por classe com o nome e probabilidade
            new_rows = []  # Lista para armazenar as novas linhas
            #for class_name, prob in zip(self.class_names, prediction_probabilities):
            # Criar a linha com a predição e probabilidade
            new_rows.append({'modelo_name': str(self.model_name), 'img_id': image_name, f'diagnostic_{self.class_names[0]}': prediction_probabilities[0], f'diagnostic_{self.class_names[1]}': prediction_probabilities[1], f'diagnostic_{self.class_names[2]}': prediction_probabilities[2], f'diagnostic_{self.class_names[3]}': prediction_probabilities[3], f'diagnostic_{self.class_names[4]}': prediction_probabilities[4], f'diagnostic_{self.class_names[5]}': prediction_probabilities[5]})

            # Concatenar as novas linhas ao DataFrame existente
            new_rows_df = pd.DataFrame(new_rows)  # Converte as novas linhas em um DataFrame
            df = pd.concat([df, new_rows_df], ignore_index=True)

            # Salvar no CSV
            df.to_csv(result_file_path, index=False)
            print(f"Dados salvos com sucesso para a imagem {image_name}!")
            return True
        except Exception as e:
            print(f"Erro ao salvar os dados: {e}")
            return False


        
    def read_csv(self):
        # Ler o arquivo CSV de metadados
        dataset = pd.read_csv(self.metadata_csv_folder_path)
        return dataset
    
    def process_image(self):
        dataset =  self.read_csv()

        # Caso já exista um dataset de predições, ignorar o processamento
        prediction_file_path = f"{self.csv_results_folder_destination}/inference_model_{self.model_name}.csv"
        if os.path.exists(prediction_file_path):
            return 
        
        for image_name in dataset['img_id']:
            loaded_image = self.get_images(image_name)
            
            if loaded_image is None:
                continue  # Pular se a imagem não for carregada
            

            # Inferência das imagens
            with torch.no_grad():
                # prediction_category, prediction_category_probability = self.inference(loaded_image)
                prediction_probabilities = self.inference(loaded_image)
            self.write_csv_file(image_name=image_name, prediction_probabilities=prediction_probabilities[0])

    def treat_one_hot_encoded(self):
        ''' Sinalizar qual o dataset foi usado como treino e validação do modelo em análise '''
        try:
            # Leitura do CSV
            dataset_pad_20_one_hot_encoded = pd.read_csv(self.csv_results_train_and_test_data, sep=",")
            print(dataset_pad_20_one_hot_encoded[["img_id", "folder"]])

            # Criação da coluna 'train'
            dataset_pad_20_one_hot_encoded["train"] = dataset_pad_20_one_hot_encoded["folder"] != self.validation_folder_number
            
            # Depois de separar os dados, drope a coluna "folder"
            dataset_pad_20_one_hot_encoded = dataset_pad_20_one_hot_encoded.drop(columns=["folder"])
            # Exibindo os dados
            return dataset_pad_20_one_hot_encoded[["img_id", "train"]]

        except Exception as e:
            print(f"Erro ao processar os dados da tabela um hot encoded. Error: {e}\n")

    def concat_dataset(self):
        try:
            # Carregar os datasets
            metadata_dataset = pd.read_csv(self.metadata_csv_folder_path, sep=",")
            predictions_dataset = pd.read_csv(os.path.join(self.csv_results_folder_destination, f"inference_model_{self.model_name}.csv"), sep=",")

            # Realizar o merge usando a coluna 'img_id' como chave
            result = pd.merge(metadata_dataset, predictions_dataset, on='img_id', how='inner')

            # Obter os dados a serem 
            dataset_pad_20_one_hot_encoded_result = self.treat_one_hot_encoded()

            # Novo merge com o dataset dos dados a serem 
            final_result = pd.merge(result, dataset_pad_20_one_hot_encoded_result, on="img_id", how="inner")

            # Salvar o dataset concatenado
            merged_file_path = os.path.join(self.csv_results_folder_destination, f"merged_metadata_folder_{self.validation_folder_number}_{self.model_name}.csv")
            final_result.to_csv(merged_file_path, index=False)
            print(f"Dados concatenados e salvos em {merged_file_path}")
            return merged_file_path
        
        except Exception as e:
            print(f"Erro ao concatenar os datasets: {e}")



if __name__=="__main__":
    pipeline = MODEL_INFERENCE(
        model_name = "mobilenet-v2",
        model_format_type = "pth",
        size_image = (224, 224),
        class_names = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
        metadata_csv_folder_path="/home/wyctor/PROJETOS/pad_ufes_20_eda/data/metadata.csv", # Caminho de onde está o metadado
        images_folder_path="/home/wyctor/PROJETOS/pad_ufes_20_eda/data/images",  # Pasta com as imagens do dataset
        validation_folder_number = "5",
        model_folder_path= "/home/wyctor/PROJETOS/pad_ufes_20_eda/src/weights/mobilenet_None_folder_5_17347249537561078/best-checkpoint/best-checkpoint.pth", #"/home/wyctor/PROJETOS/pad_ufes_20_eda/src/weights/mobile-net-cv-p5/1-mobilenet.onnx", # Caminho do modelo a ser usado
        csv_results_folder_destination="/home/wyctor/PROJETOS/pad_ufes_20_eda/src/results/inference-results/", # Onde o arquivo com os resultados das inferências será salvo
        csv_results_train_and_test_data = "/home/wyctor/PROJETOS/pad_ufes_20_eda/data/pad-ufes-20_folders_one_hot.csv"
    )
    # Processamento das imagens
    pipeline.process_image()
    # Concatenar os datasets
    pipeline.concat_dataset()
    #pipeline.treat_one_hot_encoded()

    

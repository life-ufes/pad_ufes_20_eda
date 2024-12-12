import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import onnx
import onnxruntime as ort
import os
import logging
from PIL import Image
import numpy as np

class MODEL_INFERENCE():
    def __init__(self, model_name, metadata_csv_folder_path, images_folder_path, model_folder_path, csv_results_folder_destination):
        self.model_name = model_name
        self.model_folder_path = model_folder_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()  # Carregar o modelo ONNX
        self.metadata_csv_folder_path = metadata_csv_folder_path
        self.images_folder_path = images_folder_path
        self.csv_results_folder_destination = csv_results_folder_destination
        
    def load_model(self):
        try:
            # Carregar o modelo ONNX
            onnx_model = onnx.load(self.model_folder_path)
            
            # Verificar se o modelo ONNX é válido
            onnx.checker.check_model(onnx_model)
            
            # Usar o ONNX Runtime para a inferência
            ort_session = ort.InferenceSession(self.model_folder_path)
            return ort_session
            
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo ONNX! Erro: {e}\n")
            raise
    
    def inference(self, image):
        try:
            # O modelo ONNX espera um formato específico de entrada, que geralmente é um numpy array
            ort_inputs = {self.model.get_inputs()[0].name: image}  # Obter o nome da entrada
            ort_outs = self.model.run(None, ort_inputs)  # Realizar a inferência com o ONNX Runtime
            
            prediction_probabilities = F.softmax(torch.from_numpy(ort_outs[0]), dim=1).numpy()[0]
            prediction =  np.argmax(prediction_probabilities) # Classe predita
            prediction_prob = np.max(prediction_probabilities) # Probabilidade da predição
            
            return prediction, prediction_prob
        
        except Exception as e:
            logging.error(f"Erro ao realizar a inferência: {e}")
            raise e
        
    def get_images(self, image_name):
        # Carregar a imagem e aplicar as transformações
        image_path = os.path.join(self.images_folder_path, image_name)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            transform = Compose([
                Resize((224, 224)),  # Ajuste o tamanho conforme necessário
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização comum
            ])
            image = transform(image).unsqueeze(0).numpy()  # Convert para numpy array e adiciona a dimensão batch
            return image
        else:
            logging.warning(f"Imagem {image_name} não encontrada!")
            return None
    
    def write_csv_file(self, prediction, probability, image_name):
        try:
            # Criar ou carregar o DataFrame
            result_file_path = os.path.join(self.csv_results_folder_destination, f"inference_model_{self.model_name}.csv")
            if not os.path.exists(result_file_path):
                df = pd.DataFrame(columns=['modelo_name', 'prediction', 'prediction_prob', 'img_id'])
            else:
                df = pd.read_csv(result_file_path)

            # Criar um DataFrame com a nova linha
            new_data = pd.DataFrame([{'modelo_name': self.model_name, 'prediction': prediction, 'prediction_prob': probability, 'img_id': image_name}])
            
            # Usar concatenação ao invés de append
            df = pd.concat([df, new_data], ignore_index=True)
            
            df.to_csv(result_file_path, index=False)
            logging.info(f"Dados salvos com sucesso para a imagem {image_name}!")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar os dados: {e}")
            return False
        
    def read_csv(self):
        # Ler o arquivo CSV de metadados
        dataset = pd.read_csv(self.metadata_csv_folder_path)
        return dataset
    
    def process_image(self):
        dataset =  self.read_csv()

        for image_name in dataset['img_id']:
            loaded_image = self.get_images(image_name)
            
            if loaded_image is None:
                continue  # Pular se a imagem não for carregada

            # Inferência das imagens
            with torch.no_grad():
                prediction_category, prediction_category_probability = self.inference(loaded_image)
            print(f"Imagem: {image_name}, Predição: {prediction_category} e Probabilidade:{prediction_category_probability}")
            # Salvar o valor da predição
            self.write_csv_file(prediction=prediction_category, probability=prediction_category_probability, image_name=image_name)

if __name__=="__main__":
    pipeline = MODEL_INFERENCE(
        model_name="1-mobilenet.onnx",  # Nome do modelo
        metadata_csv_folder_path="/home/wyctor/PROJETOS/pad_ufes_20_eda/data/metadata.csv", # Caminho de onde está o metadado
        images_folder_path="/home/wyctor/PROJETOS/pad_ufes_20_eda/data/images",  # Pasta com as imagens do dataset
        model_folder_path="/home/wyctor/PROJETOS/pad_ufes_20_eda/src/weights/mobile-net-cv-p5/1-mobilenet.onnx", # Caminho do modelo a ser usado
        csv_results_folder_destination="/home/wyctor/PROJETOS/pad_ufes_20_eda/src/results/inference-results/" # Onde o arquivo com os resultados das inferências será salvo
    )
    pipeline.process_image()

    

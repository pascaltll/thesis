import re
import razdel
import pandas as pd
from sklearn.model_selection import train_test_split
import torch  # Importar torchfrom torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from isanlp.pipeline_common import PipelineCommon
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from razdel import sentenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
from transformers import logging
from transformers import AutoTokenizer
logging.set_verbosity_error()


class TrainingConfig:
    def __init__(self,
                 model_name,
                 max_length=128,
                 batch_size=64,
                 epochs=3,
                 learning_rate=2e-5,
                 num_repeats=6,
                 test_size=0.2,
                 threshold=0.5):
        """
        Configuración para el entrenamiento del modelo.
        
        Args:
            model_name (str): Nombre del modelo preentrenado
            max_length (int): Longitud máxima de las secuencias
            batch_size (int): Tamaño del batch
            epochs (int): Número de épocas
            learning_rate (float): Tasa de aprendizaje
            num_repeats (int): Número de repeticiones con diferentes seeds
            test_size (float): Proporción del conjunto de prueba
            threshold (float): Umbral de similitud
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_repeats = num_repeats
        self.test_size = test_size
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)#creo que hay que usar el tokenizer de ruso
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Ahora debería funcionar


def cargar_datos(archivo, etiqueta):
    with open(archivo, 'r', encoding='utf-8') as f:
        textos = f.readlines()
    return [(texto.strip(), etiqueta) for texto in textos]

def tokenizar_fuction(texto):
    """Tokeniza el texto eliminando puntuación y convirtiendo a minúsculas."""
    texto_preprocesado = re.sub(r'[^\w\s]', '', texto.lower())
    tokens = nltk.word_tokenize(texto_preprocesado)
    return tokens

def calcular_similitud_mayoria_optimizado(textos1, textos2, threshold):
    """
        Calcula similitudes entre dos listas de textos usando GPU.
        Args:
            textos1: Lista de textos (original_datos_procesados).
            textos2: Lista de textos (train_data).
        Returns:
            Matriz booleana [len(textos1), len(textos2)] indicando similitudes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Usando dispositivo: {device}")

    # Preprocesar todos los textos en batch
    textos1_preprocesados = [' '.join(tokenizar_fuction(t)) for t in textos1]
    textos2_preprocesados = [' '.join(tokenizar_fuction(t)) for t in textos2]

    # Crear representación binaria de tokens
    vectorizer = CountVectorizer(binary=True, tokenizer=nltk.word_tokenize)
    vectorizer.fit(textos1_preprocesados + textos2_preprocesados)

    # Convertir a matrices de tokens
    matriz_tokens1 = vectorizer.transform(textos1_preprocesados).toarray()
    matriz_tokens2 = vectorizer.transform(textos2_preprocesados).toarray()

    # Mover a GPU
    matriz_tokens1 = torch.from_numpy(matriz_tokens1).to(device)
    matriz_tokens2 = torch.from_numpy(matriz_tokens2).to(device)

     # Calcular intersecciones
    interseccion = torch.matmul(matriz_tokens1.float(), matriz_tokens2.T.float())

    # Tamaños de los conjuntos de tokens
    tamano_tokens1 = matriz_tokens1.sum(dim=1).unsqueeze(1)
    tamano_tokens2 = matriz_tokens2.sum(dim=1).unsqueeze(0)

    # Evitar división por cero
    tamano_tokens1 = tamano_tokens1.clamp(min=1)
    tamano_tokens2 = tamano_tokens2.clamp(min=1)

     # Calcular proporciones
    prop_1_en_2 = interseccion / tamano_tokens2
    prop_2_en_1 = interseccion / tamano_tokens1

    similitud = (prop_1_en_2 > threshold) | (prop_2_en_1 > threshold)





    return similitud

class DataLoader_raw:
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2
        self.path_original1 = '../dataset/Первый_жанр_исходная.txt'
        self.path_original2 = '../dataset/Второй_жанр_исходная.txt'    
    
    def __call__(self):
        datos_genero1 = cargar_datos(self.path1, 0)
        datos_genero2 = cargar_datos(self.path2, 1)
        datos = datos_genero1 + datos_genero2
        
        datos_original1 = cargar_datos(self.path_original1, 0)
        datos_original2 = cargar_datos(self.path_original2, 1)
        datos_originales = datos_original1 + datos_original2
        
        return {
            'datos_raw': datos,
            'original_datos_raw': datos_originales
        }

class SentenceSplitterAndCleaner:
    def __init__(self, tokenizer, min_length_threshold=6):
        """Inicializa el procesador con un tokenizador y un umbral de longitud mínima."""
        self.tokenizer = tokenizer
        self.min_length_threshold = min_length_threshold
    
    def _clean_and_split_text(self, texto):
        """
        Limpia y divide un texto en oraciones procesadas.
        
        Args:
            texto (str): Texto crudo a procesar.
        
        Returns:
            list: Lista de oraciones limpias.
        """
        # Paso 1: Marcar puntos para evitar fusiones no deseadas
        texto_limpio = re.sub(r'\.,', '. Ok999999999 ,', texto)  # Caso 1: Punto seguido de coma
        texto_limpio = re.sub(r'\.;', '. Ok999999999 ', texto_limpio)  # Caso 2: Punto seguido de punto y coma
        texto_limpio = re.sub(r'\. ([a-zа-я])', r'. Ok999999999 \1', texto_limpio)  # Caso 3: Punto seguido de minúscula
        texto_limpio = re.sub(r'(\w)([А-Я])', r'\1. \2', texto_limpio)  # Caso 4: Minúscula seguida de mayúscula
        
        #creo que hay que agregar una nueva condivion  дл.,0 
        #", 2-3 см шир.",0 
        #", 1 см шир.",0 si hay un punto uego numeros quitar el punto
        
        # Paso 2: Eliminar corchetes y dividir en oraciones
        texto_sin_corchetes = re.sub(r'\[.*?\]', '', texto_limpio).strip()
        oraciones = [oracion.text for oracion in sentenize(texto_sin_corchetes)]
        
        # Paso 3: Restaurar espacios y limpiar marcadores temporales
        oraciones_limpias = [re.sub(r'\s*Ok999999999', ' ', oracion).strip() for oracion in oraciones]
        
        return oraciones_limpias
    
    def _process_sentences(self, datos, target_list):
        """
        Procesa un conjunto de datos y agrega oraciones válidas a la lista objetivo.
        
        Args:
            datos (list): Lista de tuplas (texto, etiqueta).
            target_list (list): Lista donde se almacenarán las oraciones procesadas.
        """
        for texto, etiqueta in datos:
            oraciones = self._clean_and_split_text(texto)
            for oracion in oraciones:
                # Filtrar oraciones cortas basadas en el número de tokens
                if len(self.tokenizer.encode(oracion, truncation=False)) >= self.min_length_threshold:
                    target_list.append((oracion, etiqueta))
    
    def __call__(self, data_raw, original_data_raw):
        """
        Procesa los datos crudos y originales, devolviendo dos conjuntos limpios.
        
        Args:
            data_raw (list): Datos modificados crudos.
            original_data_raw (list): Datos originales crudos.
        
        Returns:
            dict: Diccionario con datos procesados y originales procesados.
        """
        datos_procesados = []
        original_datos_procesados = []
        
        # Procesar datos modificados
        self._process_sentences(data_raw, datos_procesados)
        
        # Procesar datos originales
        self._process_sentences(original_data_raw, original_datos_procesados)
        
        return {
            'datos_procesados': datos_procesados,
            'original_datos_procesados': original_datos_procesados
        }

class DataProcessor:
    #def __init__(self, tokenizer, max_length, random_state, test_size=0.2):
    def __init__(self,
                 tokenizer,
                 max_length,
                 random_state,
                 test_size=0.2,
                 name='data_set_name',
                 threshold = 0.5
                ):
    
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state
        self.name = name
        self.threshold = threshold
    
    def __call__(self, datos_procesados, original_datos_procesados):
        #datos_procesados = data['datos_procesados']
        df = pd.DataFrame(datos_procesados, columns=["text", "label"])


        train_data, test_data = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        #generar un nuevo test data
        #print("test_data:", len(test_data))
        train_texts = train_data["text"].tolist()
        original_texts = [oracion for oracion, _ in original_datos_procesados]
        original_labels = [etiqueta for _, etiqueta in original_datos_procesados]
        
        similitudes = calcular_similitud_mayoria_optimizado(original_texts,
                                                            train_texts,
                                                            self.threshold)  # [len(original), len(train)]
        
        interseccion = similitudes.any(dim=1)  # True si hay intersección con algún train_text
        mask = ~interseccion.cpu().numpy()
        nuevo_test = [(texto, etiqueta) for texto, etiqueta, keep in zip(original_texts, original_labels, mask) if keep]
        #print("train_data:", len(train_data))
        #print("test_data:", len(test_data))
        #print("new_test:", len(nuevo_test))
        #print("teorical len:", len(original_texts) - len(train_data))

        
         
        if self.random_state == 0:
            train_data.to_csv(f'сокращение по частотности/train_{self.name}.csv', index=False)
            print(f"train_{self.name}.csv")
            nuevo_test_df = pd.DataFrame(nuevo_test, columns=["text", "label"])
            
            nuevo_test_df.to_csv(f'сокращение по частотности/test_{self.name}.csv', index=False)
            print(f"test_{self.name}.csv")
        
        #no se a niandido el nuevo test
        # Tokenizar entrenamiento
        train_encodings = self.tokenizer(
            train_data["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        train_labels = torch.tensor(train_data["label"].values)
        # Tokenizar prueba
        test_encodings = self.tokenizer(
            [texto for texto, _ in nuevo_test],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        test_labels = torch.tensor([etiqueta for _, etiqueta in nuevo_test])
        return {
            'train_encodings': train_encodings,
            'train_labels': train_labels,
            'test_encodings': test_encodings,
            'test_labels': test_labels
        }  


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

class DatasetCreator:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
    
    def __call__(self, train_encodings, train_labels, test_encodings, test_labels):
        train_dataset = TextDataset(train_encodings, train_labels)
        test_dataset = TextDataset(test_encodings, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return {'train_loader': train_loader, 'test_loader': test_loader}

def train_model(model, train_loader, optimizer, loss_fn, device, epochs=3):
    for epoch in range(epochs):
        model.train()
        #loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in train_loader:
        #for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            #loop.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()#libera memroria de la gpu
        gc.collect()#limpa la cpu

# +


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=["Género 1", "Género 2"])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return {"accuracy": accuracy, "report": report, "conf_matrix": conf_matrix}


# -

def display_results(results):
    print(f'Precisión en el conjunto de prueba: {results["accuracy"]:.2%}')
    print("\nReporte de clasificación:\n", results["report"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(results["conf_matrix"], annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Género 1", "Género 2"], 
                yticklabels=["Género 1", "Género 2"])
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()


def train_and_evaluate_dataset(path1, path2, config, dataset_name):
    """
    Entrena y evalúa el modelo para un conjunto de datos específico.
    
    Args:
        path1 (str): Ruta al archivo del primer género
        path2 (str): Ruta al archivo del segundo género
        config (TrainingConfig): Configuración de entrenamiento
        dataset_name (str): Nombre del conjunto de datos
    
    Returns:
        dict: Resultados con accuracy promedio y desviación estándar
    """
    seeds = list(range(config.num_repeats))
    accuracies = []
    
    #print(f"\nProcesando conjunto de datos: {dataset_name}")
    
    for seed in seeds:
        #print(f"\nRepetición con semilla {seed}")
        
        # Crear pipeline
        ppl = PipelineCommon([
            (DataLoader_raw(path1, path2), 
             [], 
             {'datos_raw': 'datos_raw',
              'original_datos_raw': 'original_datos_raw'}),
            (SentenceSplitterAndCleaner(config.tokenizer), 
             ['datos_raw', 'original_datos_raw'], 
             {'datos_procesados': 'datos_procesados',
              'original_datos_procesados': 'original_datos_procesados'}),
            (DataProcessor(config.tokenizer,
                          config.max_length,
                          seed,
                          name=dataset_name,
                          threshold=config.threshold), 
             ['datos_procesados', 'original_datos_procesados'], 
             {'train_encodings': 'train_encodings', 
              'train_labels': 'train_labels',
              'test_encodings': 'test_encodings', 
              'test_labels': 'test_labels'}),
            (DatasetCreator(batch_size=config.batch_size), 
             ['train_encodings', 'train_labels', 
              'test_encodings', 'test_labels'], 
             {'train_loader': 'train_loader', 
              'test_loader': 'test_loader'})
        ])
        
        # Ejecutar pipeline
        result = ppl()
        train_loader = result['train_loader']
        test_loader = result['test_loader']
        
        # Configurar modelo
        model = BertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2).to(config.device)
        
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Entrenar y evaluar
        train_model(model, train_loader, optimizer, loss_fn, 
                   config.device, epochs=config.epochs)
        results = evaluate_model(model, test_loader, config.device)
        
        accuracy = results['accuracy']
        accuracies.append(accuracy)
        #print(f"Accuracy con semilla {seed}: {accuracy:.4f}")
    
    # Calcular estadísticas
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    #print(f"\nAccuracy promedio para {dataset_name}: " f"{avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return {
        'dataset_name': dataset_name,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'accuracies': accuracies
    }




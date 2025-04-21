# +
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, pipeline
from tqdm import tqdm
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from razdel import sentenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from transformers import logging
from transformers import AutoTokenizer
logging.set_verbosity_error()

import matplotlib.pyplot as plt
import numpy as np


nltk.download('punkt_tab')
nltk.download('punkt')

# -

class TrainingConfig:
    def __init__(self,
                 model_name,
                 max_length=128,
                 batch_size=64,
                 epochs=3,
                 learning_rate=2e-5,
                 num_repeats=6,
                 test_size=0.2,
                 threshold=0.5,
                 model_type='bert'):
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
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)#creo que hay que usar el tokenizer de ruso
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        if self.model_type == 'gpt':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'



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
    def __init__(self,
                 tokenizer,
                 max_length,
                 random_state,
                 test_size=0.2,
                 name='data_set_name',
                 threshold=0.5,
                 model_type='bert'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state
        self.name = name
        self.threshold = threshold
        self.model_type = model_type
        
    def __call__(self, datos_procesados, original_datos_procesados):
        df = pd.DataFrame(datos_procesados, columns=["text", "label"])
        train_data, test_data = train_test_split(df,
                                                 test_size=self.test_size,
                                                 random_state=self.random_state)
        
        if self.model_type == 'gpt':
            train_texts = train_data["text"].tolist()
            train_labels = train_data["label"].tolist()
            
            original_texts = [oracion for oracion, _ in original_datos_procesados]
            original_labels = [etiqueta for _, etiqueta in original_datos_procesados]
            
            # Calcular similitudes para filtrar el conjunto de prueba
            similitudes = calcular_similitud_mayoria_optimizado(original_texts, train_texts, self.threshold)
            interseccion = similitudes.any(dim=1)
            mask = ~interseccion.cpu().numpy()
            nuevo_test = [(texto, etiqueta) for texto, etiqueta, keep in zip(original_texts, original_labels, mask) if keep]
            
            # Tokenizar entrenamiento
            train_encodings = self.tokenizer(
                train_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            train_labels = torch.tensor(train_labels)
            
            # Tokenizar prueba
            test_encodings = self.tokenizer(
                [texto for texto, _ in nuevo_test],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            test_labels = torch.tensor([etiqueta for _, etiqueta in nuevo_test])
            
            # Guardar datos si random_state == 0
            if self.random_state == 0:
                train_data.to_csv(f'сокращение по частотности/train_{self.name}.csv', index=False)
                print(f"train_{self.name}.csv")
                nuevo_test_df = pd.DataFrame(nuevo_test, columns=["text", "label"])
                nuevo_test_df.to_csv(f'сокращение по частотности/test_{self.name}.csv', index=False)
                print(f"test_{self.name}.csv")
            
        else:
            train_texts = train_data["text"].tolist()
            original_texts = [oracion for oracion, _ in original_datos_procesados]
            original_labels = [etiqueta for _, etiqueta in original_datos_procesados]

            similitudes = calcular_similitud_mayoria_optimizado(original_texts,
                                                                train_texts,
                                                                self.threshold)
            interseccion = similitudes.any(dim=1)
            mask = ~interseccion.cpu().numpy()
            nuevo_test = [(texto, etiqueta) for texto, etiqueta, keep in zip(original_texts, original_labels, mask) if keep]

            if self.random_state == 0:
                train_data.to_csv(f'сокращение по частотности/train_{self.name}.csv', index=False)
                print(f"train_{self.name}.csv")
                nuevo_test_df = pd.DataFrame(nuevo_test, columns=["text", "label"])
                nuevo_test_df.to_csv(f'сокращение по частотности/test_{self.name}.csv', index=False)
                print(f"test_{self.name}.csv")

            # Tokenizar entrenamiento
            train_encodings = self.tokenizer(
                train_data["text"].tolist(),
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            train_labels = torch.tensor(train_data["label"].values)
            test_encodings = self.tokenizer(
                [texto for texto, _ in nuevo_test],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            test_labels = torch.tensor([etiqueta for _, etiqueta in nuevo_test])
        
        # Depuración: Verificar forma y contenido de las etiquetas
        print(f"Train labels sample: {train_labels[:5]}, Shape: {train_labels.shape}")
        print(f"Test labels sample: {test_labels[:5]}, Shape: {test_labels.shape}")
        
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

def train_model(model, train_loader, optimizer, loss_fn, device, epochs=3, 
                model_type='bert', dataset_name=None, model_name=None, classifier=None):
    """
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader para entrenamiento
        optimizer: Optimizador
        loss_fn: Función de pérdida
        device: Dispositivo (cuda/cpu)
        epochs: Número de épocas
        model_type: Tipo de modelo ('bert' o 'gpt')
        dataset_name: Nombre del dataset para logging
        model_name: Nombre del modelo para logging
        classifier: Clasificador lineal para GPT (opcional)
    """
    model.train()
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if model_type == 'gpt':
                if classifier is None:
                    raise ValueError("Classifier must be provided for GPT model")
                
                # Obtener hidden states para GPT
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Tomar el último layer de hidden states [batch_size, seq_len, hidden_size]
                last_hidden = outputs.hidden_states[-1]
                # Promedio de todos los tokens (consistente con evaluate_model)
                pooled = last_hidden.mean(dim=1)
                
                # Pasar por el clasificador lineal
                logits = classifier(pooled)
                
                # Calcular pérdida
                loss = loss_fn(logits, labels)
            else:
                # Entrenamiento original para BERT
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
            
            # Log batch loss (mantener si usas WandB u otro sistema de logging)
            
        epoch_loss = total_loss / total_batches
        epoch_losses.append(epoch_loss)
        
        # Log epoch metrics (mantener si usas WandB u otro sistema de logging)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return epoch_loss  # Return final epoch loss for summary


# +
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, log_loss
import torch.nn.functional as F
import numpy as np

def evaluate_model(model, test_loader, device, tokenizer, model_type='bert', classifier=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Lista para acumular probabilidades por lote
    
    if model_type == 'gpt':
        if classifier is None:
            raise ValueError("Classifier must be provided for GPT model")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                true_labels = batch['labels'].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                pooled = last_hidden.mean(dim=1)
                logits = classifier(pooled)
                probs = F.softmax(logits, dim=1).cpu().numpy()  # [batch_size, 2]
                pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(pred_labels)
                all_labels.extend(true_labels)
                all_probs.append(probs)  # Acumular array por lote
    else:
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()  # [batch_size, 2]
                predictions = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs)  # Acumular array por lote
    
    # Convertir listas a arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)  # Concatenar arrays en [num_samples, 2]
    
    # Depuración: Verificar formas
    print(f"all_preds shape: {all_preds.shape}")
    print(f"all_labels shape: {all_labels.shape}")
    print(f"all_probs shape: {all_probs.shape}")
    
    # Calcular métricas
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=["жанр0", "жанр1"], output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])  # Probabilidad de clase 1
    pr_auc = average_precision_score(all_labels, all_probs[:, 1])
    log_loss_val = log_loss(all_labels, all_probs)
    
    return {
        "accuracy": accuracy,
        "f1_weighted": report['weighted avg']['f1-score'],
        "f1_macro": report['macro avg']['f1-score'],
        "f1_class0": report['жанр0']['f1-score'],
        "f1_class1": report['жанр1']['f1-score'],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": log_loss_val,
        "report": report,
        "conf_matrix": conf_matrix,
        "predictions": all_preds.tolist(),
        "true_labels": all_labels.tolist(),
        "probs": all_probs.tolist()
    }


# -

def train_and_evaluate_dataset(path1, path2, config, dataset_name, dataset_type):
    seeds = list(range(config.num_repeats))
    accuracies = []
    f1_weighteds = []
    f1_macros = []
    f1_class0s = []
    f1_class1s = []
    roc_aucs = []
    pr_aucs = []
    log_losses = []
    losses = []
    
    for seed in seeds:
        # Pipeline común
        ppl = PipelineCommon([
            (DataLoader_raw(path1, path2), [], {'datos_raw': 'datos_raw', 'original_datos_raw': 'original_datos_raw'}),
            (SentenceSplitterAndCleaner(config.tokenizer), ['datos_raw', 'original_datos_raw'], 
             {'datos_procesados': 'datos_procesados', 'original_datos_procesados': 'original_datos_procesados'}),
            (DataProcessor(config.tokenizer, config.max_length, seed, name=dataset_name, 
                           threshold=config.threshold, model_type=config.model_type), 
             ['datos_procesados', 'original_datos_procesados'], 
             {'train_encodings': 'train_encodings', 'train_labels': 'train_labels',
              'test_encodings': 'test_encodings', 'test_labels': 'test_labels'}),
            (DatasetCreator(batch_size=config.batch_size), 
             ['train_encodings', 'train_labels', 'test_encodings', 'test_labels'], 
             {'train_loader': 'train_loader', 'test_loader': 'test_loader'})
        ])
        
        result = ppl()
        train_loader = result['train_loader']
        test_loader = result['test_loader']
        
        # Configurar modelo
        if config.model_type == 'gpt':
            model = AutoModelForCausalLM.from_pretrained(config.model_name, output_hidden_states=True).to(config.device)
            classifier = torch.nn.Linear(model.config.hidden_size, 2).to(config.device)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=2).to(config.device)
            classifier = None
            loss_fn = torch.nn.CrossEntropyLoss()
        
        optimizer = AdamW(list(model.parameters()) + (list(classifier.parameters()) if classifier else []), 
                         lr=config.learning_rate)
        
        final_loss = train_model(model, train_loader, optimizer, loss_fn, config.device, 
                                epochs=config.epochs, model_type=config.model_type, 
                                dataset_name=dataset_name, model_name=config.model_name, classifier=classifier)
        losses.append(final_loss)
        
        results = evaluate_model(model, test_loader, config.device, config.tokenizer, 
                               config.model_type, classifier=classifier)
        
        accuracies.append(results['accuracy'])
        f1_weighteds.append(results['f1_weighted'])
        f1_macros.append(results['f1_macro'])
        f1_class0s.append(results['f1_class0'])
        f1_class1s.append(results['f1_class1'])
        roc_aucs.append(results['roc_auc'])
        pr_aucs.append(results['pr_auc'])
        log_losses.append(results['log_loss'])
        
        # Opcional: Log en WandB
        # import wandb
        # wandb.log({
        #     "seed": seed,
        #     "accuracy": results['accuracy'],
        #     "f1_weighted": results['f1_weighted'],
        #     "f1_macro": results['f1_macro'],
        #     "f1_class0": results['f1_class0'],
        #     "f1_class1": results['f1_class1'],
        #     "roc_auc": results['roc_auc'],
        #     "pr_auc": results['pr_auc'],
        #     "log_loss": results['log_loss'],
        #     "train_loss": final_loss,
        #     "confusion_matrix": wandb.plot.confusion_matrix(
        #         y_true=results['true_labels'], 
        #         preds=results['predictions'], 
        #         class_names=["жанр0", "жанр1"]
        #     )
        # })
    
    return {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'model_name': config.model_name,
        'avg_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'avg_f1_weighted': np.mean(f1_weighteds),
        'std_f1_weighted': np.std(f1_weighteds),
        'avg_f1_macro': np.mean(f1_macros),
        'std_f1_macro': np.std(f1_macros),
        'avg_f1_class0': np.mean(f1_class0s),
        'std_f1_class0': np.std(f1_class0s),
        'avg_f1_class1': np.mean(f1_class1s),
        'std_f1_class1': np.std(f1_class1s),
        'avg_roc_auc': np.mean(roc_aucs),
        'std_roc_auc': np.std(roc_aucs),
        'avg_pr_auc': np.mean(pr_aucs),
        'std_pr_auc': np.std(pr_aucs),
        'avg_log_loss': np.mean(log_losses),
        'std_log_loss': np.std(log_losses),
        'avg_loss': np.mean(losses),
        'accuracies': accuracies,
        'f1_weighteds': f1_weighteds,
        'f1_macros': f1_macros,
        'f1_class0s': f1_class0s,
        'f1_class1s': f1_class1s,
        'roc_aucs': roc_aucs,
        'pr_aucs': pr_aucs,
        'log_losses': log_losses,
        'losses': losses,
        'type': 'gpt' if config.model_type == 'gpt' else 'bert'
    }


# +
def plot_model_performance(datasets, models, results, save_path='frecuencia_profesional.png'):
    """
    Genera un gráfico de barras comparando el rendimiento de modelos por dataset.
    
    Args:
        datasets (list): Lista de diccionarios con nombres de datasets (cada uno con clave 'name').
        models (list): Lista de diccionarios con nombres de modelos (cada uno con clave 'name').
        results (list): Lista de resultados con 'dataset_name', 'model_name' y 'avg_accuracy'.
        save_path (str): Ruta donde guardar el gráfico (por defecto 'frecuencia_profesional.png').
    """
    # Extraer nombres
    dataset_names = [d['name'] for d in datasets]
    model_names = [m['name'] for m in models]
    
    # Crear matriz de precisiones
    accuracies = np.zeros((len(dataset_names), len(model_names)))
    for i, dataset_name in enumerate(dataset_names):
        for j, model_name in enumerate(model_names):
            for result in results:
                if dataset_name in result['dataset_name'] and model_name in result['dataset_name']:
                    accuracies[i, j] = result['avg_accuracy']
                    break  # Salir del bucle una vez encontrada la precisión
    
    # Configuración de estilo profesional
    plt.style.use('seaborn-v0_8')
    
    # Crear figura con mayor resolución
    #fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    fig, ax = plt.subplots(figsize=(20, 8), dpi=100)
    
    # Parámetros dinámicos según cantidad de modelos
    n_models = len(model_names)
    bar_width = min(0.8 / n_models, 0.15)  # Ajustar ancho de barras dinámicamente
    index = np.arange(len(dataset_names))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))  # Paleta de colores profesional
    
    # Crear barras con mejoras visuales
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        bars = plt.bar(index + i * bar_width,
                       accuracies[:, i],
                       bar_width,
                       label=model_name,
                       color=color,
                       edgecolor='black',
                       alpha=0.8)
        
        # Añadir valores encima de cada barra solo si no hay demasiados modelos
        if n_models <= 10:  # Límite para evitar clutter
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Personalización de ejes
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Precisión Promedio', fontsize=12, fontweight='bold')
    plt.title('Rendimiento de Modelos por Dataset', fontsize=14, fontweight='bold', pad=20)
    
    # Ajustar marcas del eje X
    plt.xticks(index + bar_width * (n_models - 1) / 2,
               dataset_names,
               rotation=45,
               ha='right',
               fontsize=10)
    
    # Añadir grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Mover la leyenda fuera del gráfico
    plt.legend(title='Modelos',
               title_fontsize=12,
               fontsize=10,
               loc='center left',
               bbox_to_anchor=(1.05, 0.5),
               frameon=True,
               borderaxespad=0.)
    
    # Ajustar límites del eje Y
    plt.ylim(0, max(accuracies.max() * 1.1, 1))
    
    # Ajustar layout para incluir la leyenda externa
    plt.tight_layout()
    
    # Guardar con alta calidad
    plt.savefig(save_path,
                dpi=300,
                bbox_inches='tight')
    
    # Mostrar gráfico
    plt.show()

# Ejemplo de uso:
# datasets = [{'name': 'dataset1'}, {'name': 'dataset2'}]
# models = [{'name': 'model1'}, {'name': 'model2'}]
# results = [{'dataset_name': 'dataset1_model1', 'avg_accuracy': 0.85}, ...]
# plot_model_performance(datasets, models, results, save_path='frecuencia_profesional.png')


# +
import json
import numpy as np
import matplotlib.pyplot as plt

def read_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_data(results):
    """Organize data for visualization"""
    dataset_names = sorted(list(set(item['dataset_name'] for item in results)))
    model_names = sorted(list(set(item['model_name'] for item in results)))
    
    accuracy_matrix = np.zeros((len(dataset_names), len(model_names)))
    
    for item in results:
        dataset_idx = dataset_names.index(item['dataset_name'])
        model_idx = model_names.index(item['model_name'])
        accuracy_matrix[dataset_idx, model_idx] = item['avg_accuracy']
    
    return dataset_names, model_names, accuracy_matrix

def create_comparison_plot(datasets, models, accuracies, title_suffix, save_path):
    """Generate a single comparison plot"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    
    n_models = len(models)
    bar_width = min(0.8 / n_models, 0.15)
    index = np.arange(len(datasets))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        bars = plt.bar(
            index + i * bar_width,
            accuracies[:, i],
            bar_width,
            label=model,
            color=color,
            edgecolor='black',
            alpha=0.8
        )
        
        if n_models <= 10:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{height:.3f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=9
                )
    
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance Comparison ({title_suffix})', fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(
        index + bar_width * (n_models - 1) / 2,
        datasets,
        rotation=45,
        ha='right',
        fontsize=10
    )
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(
        title='Models',
        title_fontsize=12,
        fontsize=10,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=True
    )
    
    plt.ylim(0, max(accuracies.max() * 1.1, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_separate_plots(dataset_names, model_names, accuracy_matrix):
    """Create two separate plots: frequency-based and other datasets"""
    # Split datasets into two groups
    freq_indices = [i for i, name in enumerate(dataset_names) if 'frequenc' in name.lower()]
    other_indices = [i for i, name in enumerate(dataset_names) if 'frequency' not in name.lower()]
    
    # Frequency-based datasets plot
#     if freq_indices:
#         freq_datasets = [dataset_names[i] for i in freq_indices]
#         freq_accuracies = accuracy_matrix[freq_indices, :]
#         create_comparison_plot(
#             freq_datasets, 
#             model_names, 
#             freq_accuracies,
#             'Frequency-based Datasets',
#             'performance_comparison_frequency.png'
#         )
    
    # Other datasets plot
    if other_indices:
        other_datasets = [dataset_names[i] for i in other_indices]
        other_accuracies = accuracy_matrix[other_indices, :]
        create_comparison_plot(
            other_datasets, 
            model_names, 
            other_accuracies,
            'Other Datasets',
            'performance_comparison_other.png'
        )

if __name__ == "__main__":
    results = read_results('model_results.json')
    dataset_names, model_names, accuracy_matrix = prepare_data(results)
    
    print("Analyzed datasets:", dataset_names)
    print("Evaluated models:", model_names)
    print("Accuracy matrix:\n", accuracy_matrix)
    
    create_separate_plots(dataset_names, model_names, accuracy_matrix)
# -



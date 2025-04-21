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

class ZeroShotConfig:
    def __init__(self,
                 model_name,
                 candidate_labels=["жанр0", "жанр1"],
                 hypothesis_template="Este texto es sobre {}."):
        self.model_name = model_name
        self.candidate_labels = candidate_labels
        self.hypothesis_template = hypothesis_template
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_shot_evaluate(config, test_loader, tokenizer):
    # Cargar el pipeline de zero-shot
    classifier = pipeline(
        "zero-shot-classification",
        model=config.model_name,
        device=config.device
    )
    
    all_true_labels = []
    all_preds = []
    all_probs = []
    
    # Procesar el test_loader
    for batch in test_loader:
        texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                for ids in batch['input_ids']]
        true_labels = batch['labels'].cpu().numpy()
        
        # Clasificación zero-shot
        results = classifier(
            texts,
            candidate_labels=config.candidate_labels,
            hypothesis_template=config.hypothesis_template
        )
        
        # Procesar resultados
        for result, true_label in zip(results, true_labels):
            pred_label = config.candidate_labels.index(result['labels'][0])
            prob = result['scores'][0]
            
            all_true_labels.append(true_label)
            all_preds.append(pred_label)
            all_probs.append(prob)
    
    # Convertir a arrays numpy
    all_true_labels = np.array(all_true_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calcular métricas (igual que en tu evaluate_model)
    accuracy = np.mean(all_preds == all_true_labels)
    report = classification_report(all_true_labels, all_preds, 
                                  target_names=config.candidate_labels, output_dict=True)
    conf_matrix = confusion_matrix(all_true_labels, all_preds)
    roc_auc = roc_auc_score(all_true_labels, all_probs)
    pr_auc = average_precision_score(all_true_labels, all_probs)
    log_loss_val = log_loss(all_true_labels, all_probs)
    
    return {
        "accuracy": accuracy,
        "f1_weighted": report['weighted avg']['f1-score'],
        "f1_macro": report['macro avg']['f1-score'],
        "f1_class0": report[config.candidate_labels[0]]['f1-score'],
        "f1_class1": report[config.candidate_labels[1]]['f1-score'],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": log_loss_val,
        "report": report,
        "conf_matrix": conf_matrix,
        "predictions": all_preds.tolist(),
        "true_labels": all_true_labels.tolist(),
        "probs": all_probs.tolist()
    }


def train_and_evaluate_zero_shot(path1, path2, config, dataset_name, dataset_type):
    seeds = list(range(config.num_repeats))
    metrics = {
        'accuracies': [],
        'f1_weighteds': [],
        'f1_macros': [],
        'f1_class0s': [],
        'f1_class1s': [],
        'roc_aucs': [],
        'pr_aucs': [],
        'log_losses': [],
        'confusion_matrices': []
    }
    
    for seed in seeds:
        # Pipeline común (igual que antes para preparar los datos)
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
        test_loader = result['test_loader']
        
        # Configuración Zero-Shot
        zs_config = ZeroShotConfig(
            model_name=config.model_name,
            candidate_labels=["жанр0", "жанр1"],
            hypothesis_template="Este fragmento literario pertenece al género {}."
        )
        
        # Evaluación Zero-Shot
        results = zero_shot_evaluate(zs_config, test_loader, config.tokenizer)
        
        # Acumular métricas
        metrics['accuracies'].append(results['accuracy'])
        metrics['f1_weighteds'].append(results['f1_weighted'])
        metrics['f1_macros'].append(results['f1_macro'])
        metrics['f1_class0s'].append(results['f1_class0'])
        metrics['f1_class1s'].append(results['f1_class1'])
        metrics['roc_aucs'].append(results['roc_auc'])
        metrics['pr_aucs'].append(results['pr_auc'])
        metrics['log_losses'].append(results['log_loss'])
        metrics['confusion_matrices'].append(results['conf_matrix'])
    
    # Calcular promedios
    avg_conf_matrix = np.mean(metrics['confusion_matrices'], axis=0)
    
    return {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'model_name': config.model_name,
        'avg_accuracy': float(np.mean(metrics['accuracies'])),
        'std_accuracy': float(np.std(metrics['accuracies'])),
        'avg_f1_weighted': float(np.mean(metrics['f1_weighteds'])),
        'std_f1_weighted': float(np.std(metrics['f1_weighteds'])),
        'avg_f1_macro': float(np.mean(metrics['f1_macros'])),
        'std_f1_macro': float(np.std(metrics['f1_macros'])),
        'avg_f1_class0': float(np.mean(metrics['f1_class0s'])),
        'std_f1_class0': float(np.std(metrics['f1_class0s'])),
        'avg_f1_class1': float(np.mean(metrics['f1_class1s'])),
        'std_f1_class1': float(np.std(metrics['f1_class1s'])),
        'avg_roc_auc': float(np.mean(metrics['roc_aucs'])),
        'std_roc_auc': float(np.std(metrics['roc_aucs'])),
        'avg_pr_auc': float(np.mean(metrics['pr_aucs'])),
        'std_pr_auc': float(np.std(metrics['pr_aucs'])),
        'avg_log_loss': float(np.mean(metrics['log_losses'])),
        'std_log_loss': float(np.std(metrics['log_losses'])),
        'avg_confusion_matrix': avg_conf_matrix.tolist(),
        'confusion_matrices': [cm.tolist() for cm in metrics['confusion_matrices']],
        'type': 'zero-shot',
        'true_labels': results['true_labels'],
        'predictions': results['predictions']
    }


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
    
    torch.cuda.empty_cache()
    gc.collect()
    
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
        "predictions": all_preds.tolist(),  # Should include predictions
        "true_labels": all_labels.tolist(),  # Should include true_labels
        "probs": all_probs.tolist()
    }


# +
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
    confusion_matrices = []
    
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
        confusion_matrices.append(results['conf_matrix'])  # Store confusion matrix
        

        


    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    if classifier is not None:
        del classifier
    torch.cuda.empty_cache()
    gc.collect()

#     return {
#         'dataset_name': dataset_name,
#         'dataset_type': dataset_type,
#         'model_name': config.model_name,
#         'avg_accuracy': np.mean(accuracies),
#         'std_accuracy': np.std(accuracies),
#         'avg_f1_weighted': np.mean(f1_weighteds),
#         'std_f1_weighted': np.std(f1_weighteds),
#         'avg_f1_macro': np.mean(f1_macros),
#         'std_f1_macro': np.std(f1_macros),
#         'avg_f1_class0': np.mean(f1_class0s),
#         'std_f1_class0': np.std(f1_class0s),
#         'avg_f1_class1': np.mean(f1_class1s),
#         'std_f1_class1': np.std(f1_class1s),
#         'avg_roc_auc': np.mean(roc_aucs),
#         'std_roc_auc': np.std(roc_aucs),
#         'avg_pr_auc': np.mean(pr_aucs),
#         'std_pr_auc': np.std(pr_aucs),
#         'avg_log_loss': np.mean(log_losses),
#         'std_log_loss': np.std(log_losses),
#         'avg_loss': np.mean(losses),
#         'accuracies': accuracies,
#         'f1_weighteds': f1_weighteds,
#         'f1_macros': f1_macros,
#         'f1_class0s': f1_class0s,
#         'f1_class1s': f1_class1s,
#         'roc_aucs': roc_aucs,
#         'pr_aucs': pr_aucs,
#         'log_losses': log_losses,
#         'losses': losses,
#         'avg_confusion_matrix': avg_confusion_matrix,  # Add average confusion matrix
#         'confusion_matrices': confusion_matrices,  # Include all confusion matrices
#         'type': 'gpt' if config.model_type == 'gpt' else 'bert',
#         'true_labels': results['true_labels'],
#         'predictions': results['predictions']
#     }
    return {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'model_name': config.model_name,
        'avg_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'avg_f1_weighted': float(np.mean(f1_weighteds)),
        'std_f1_weighted': float(np.std(f1_weighteds)),
        'avg_f1_macro': float(np.mean(f1_macros)),
        'std_f1_macro': float(np.std(f1_macros)),
        'avg_f1_class0': float(np.mean(f1_class0s)),
        'std_f1_class0': float(np.std(f1_class0s)),
        'avg_f1_class1': float(np.mean(f1_class1s)),
        'std_f1_class1': float(np.std(f1_class1s)),
        'avg_roc_auc': float(np.mean(roc_aucs)),
        'std_roc_auc': float(np.std(roc_aucs)),
        'avg_pr_auc': float(np.mean(pr_aucs)),
        'std_pr_auc': float(np.std(pr_aucs)),
        'avg_log_loss': float(np.mean(log_losses)),
        'std_log_loss': float(np.std(log_losses)),
        'avg_loss': float(np.mean(losses)),
        'accuracies': [float(x) for x in accuracies],
        'f1_weighteds': [float(x) for x in f1_weighteds],
        'f1_macros': [float(x) for x in f1_macros],
        'f1_class0s': [float(x) for x in f1_class0s],
        'f1_class1s': [float(x) for x in f1_class1s],
        'roc_aucs': [float(x) for x in roc_aucs],
        'pr_aucs': [float(x) for x in pr_aucs],
        'log_losses': [float(x) for x in log_losses],
        'losses': [float(x) for x in losses],
        'avg_confusion_matrix': avg_confusion_matrix.tolist(),  # convert ndarray to list
        'confusion_matrices': [cm.tolist() for cm in confusion_matrices],  # list of ndarrays
        'type': 'gpt' if config.model_type == 'gpt' else 'bert',
        'true_labels': results['true_labels'],  # already lists
        'predictions': results['predictions']   # already lists
    }

# -




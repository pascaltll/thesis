import funciones
from utils import train_wrapper

import warnings
import os
# Suprimir warnings específicos
warnings.filterwarnings('ignore', category=UserWarning)  # Para sklearn y otros
warnings.filterwarnings('ignore', category=FutureWarning)  # Para huggingface y transformers
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from contextlib import redirect_stderr
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

def train_wrapper(args):
    dataset, config = args  # Desempaquetar los argumentos
    return funciones.train_and_evaluate_dataset(
        dataset['path1'],
        dataset['path2'],
        config,
        dataset['name']
    )

# Uso del código
def main():
    # Crear configuración
    config = funciones.TrainingConfig(
        model_name='DeepPavlov/rubert-base-cased',
        max_length=128,
        batch_size=64,
        epochs=3,
        learning_rate=2e-5,
        num_repeats=6,
        test_size=0.2,
        threshold=0.5
    )
    
    # Lista de conjuntos de datos a procesar
    
    datasets = [
        {
            'path1': '../dataset/сокращение по частотности/1б_Изъяты лексемы с частотой выше 100.txt',
            'path2': '../dataset/сокращение по частотности/2б_Изъяты лексемы с частотой выше 100.txt',
            'name': 'Изъяты лексемы с частотой выше 100'
        },
        {
            'path1': '../dataset/сокращение по частотности/1в_Изъяты лексемы с частотой выше 49.txt',
            'path2': '../dataset/сокращение по частотности/2в_Изъяты лексемы с частотой выше 49.txt',
            'name': 'Изъяты лексемы с частотой выше 49'
        },
        {
            'path1': '../dataset/сокращение по частотности/1г_Изъяты лексемы с частотой выше 29.txt',
            'path2': '../dataset/сокращение по частотности/2г_Изъяты лексемы с частотой выше 29.txt',
            'name': 'Изъяты лексемы с частотой выше 29'
        },
        {
            'path1': '../dataset/сокращение по частотности/1д_Изъяты лексемы с частотой выше 9.txt',
            'path2': '../dataset/сокращение по частотности/2д_Изъяты лексемы с частотой выше 9.txt',
            'name': 'Изъяты лексемы с частотой выше 9'
        },
        {
            'path1': '../dataset/сокращение по частотности/1е_Изъяты лексемы с частотой выше 5.txt',
            'path2': '../dataset/сокращение по частотности/2е_Изъяты лексемы с частотой выше 5.txt',
            'name': 'Изъяты лексемы с частотой выше 5'
        },
        {
            'path1': '../dataset/сокращение по частотности/1ё_Изъяты лексемы с частотой выше 3.txt',
            'path2': '../dataset/сокращение по частотности/2ё_Изъяты лексемы с частотой выше 3.txt',
            'name': 'Изъяты лексемы с частотой выше 3'
        },
        
        
    ]
    
    # Procesar todos los conjuntos de datos
    results = []
    #for dataset in datasets:
#     for dataset in funciones.tqdm(datasets, total=len(datasets)):
#         result = funciones.train_and_evaluate_dataset(
#             dataset['path1'],
#             dataset['path2'],
#             config,
#             dataset['name']
#         )
#         results.append(result)
    

    args = [(dataset, config) for dataset in datasets]
 
    # Procesar datasets en paralelo
    result = []
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        with ProcessPoolExecutor() as executor:
            result = list(tqdm(
                executor.map(train_wrapper, args),
                total = len(datasets),
                desc = "train in parallel"
                ))
            
            
    
    # Mostrar resumen final
    print("\nResumen de todos los experimentos:")
    for result in results:
        print(f"{result['dataset_name']}: "
              f"{result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f}")

        
if __name__ == "__main__":
    main()

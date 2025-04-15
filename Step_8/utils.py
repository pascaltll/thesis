# utils.py
import funciones

def train_wrapper(args):
    dataset, config = args
    return funciones.train_and_evaluate_dataset(
        dataset['path1'],
        dataset['path2'],
        config,
        dataset['name']
    )

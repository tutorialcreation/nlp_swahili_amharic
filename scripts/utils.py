from mlflow import get_experiment_by_name
import os

def get_experiment_id(file):
    current_experiment=dict(get_experiment_by_name("abtest"))
    experiment_id=current_experiment['experiment_id']
    file_path = os.path.abspath(os.path.join(f'mlruns/0/{experiment_id}/artifacts/{file}')) 
    return file_path
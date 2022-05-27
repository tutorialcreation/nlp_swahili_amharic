from this import d
import mlflow
import pickle
import time
import joblib
from scripts.logger import logger

class ModelSerializer:
    """
    - this class is responsible for
    serializing models

    """ 
    
    def __init__(self,model) -> None:
        self.model = model
        logger.info("initializing serializer")

    def pickle_serialize(self):
        """
        - algorithm for serializing the models
        """
        file_name =  time.strftime("%Y%m%d-%H%M%S")
        with open(f'models/{file_name}.pkl', 'wb') as files:
            pickle.dump(self.model, files)
        
        mlflow.log_artifact(f"models/{file_name}.pkl")
        logger.info("Successfully saved the model")

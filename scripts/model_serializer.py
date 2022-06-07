from this import d
import mlflow
import pickle
import time
import joblib

import os,sys
# sys.path.append(os.path.abspath(os.path.join('../scripts')))
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
        try:
            with open(f'../models/{file_name}.pkl', 'wb') as files:
                pickle.dump(self.model, files)

            with open(f'../models/model.pkl', 'wb') as files:
                pickle.dump(self.model, files)
        except:
            try:
                with open(f'models/{file_name}.pkl', 'wb') as files:
                    pickle.dump(self.model, files)

                with open(f'models/model.pkl', 'wb') as files:
                    pickle.dump(self.model, files)
            except Exception as e:
                logger.error(e)
            
        mlflow.log_artifact(f"models/{file_name}.pkl")
        logger.info("Successfully saved the model")

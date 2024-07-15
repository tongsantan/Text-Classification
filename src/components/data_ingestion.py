## 5. Update the components

import os
import sys
from src.exception import CustomException
from src import logger
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.utils.common import encoded
import numpy as np
from src.entity.config_entity import DataIngestionConfig

## 5. Update the components

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        '''
        This function is responsible for data ingestion
        
        '''
        logger.info("Data Ingestion") 
        
        try:
            logger.info("Reading the data")
            df=pd.read_csv(self.config.input_data_path)
            
            df = df[['TITLE','CATEGORY']]
            
            logger.info("Sampling 5000 rows for each category at random") 
            e = df[df['CATEGORY'] == 'e'].sample(n=5000)
            b = df[df['CATEGORY'] == 'b'].sample(n=5000)
            t = df[df['CATEGORY'] == 't'].sample(n=5000)
            m = df[df['CATEGORY'] == 'm'].sample(n=5000)

            df_selected = pd.concat([e,b,t,m], ignore_index=True)
            df_selected = df_selected.reindex(np.random.permutation(df_selected.index)).reset_index(drop=True)
            df_selected['TARGET'] = df_selected.apply(lambda x: encoded(x['CATEGORY']), axis=1)
            
            os.makedirs(os.path.dirname(self.config.processed_data_path),exist_ok=True)
            df_selected.to_csv(self.config.processed_data_path,index=False,header=True)   
            
            logger.info("Splitting the data into X and y")
            seed = 18
            X = df_selected['TITLE']
            y = df_selected['TARGET']
            
            logger.info("Train test split initiated to create Training, Validation and Test data")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=seed)
            
            logger.info("Saving the data")
            os.makedirs(os.path.dirname(self.config.X_train_data_path),exist_ok=True)
            
            X_train.to_csv(self.config.X_train_data_path,index=False,header=True)
            X_test.to_csv(self.config.X_test_data_path,index=False,header=True)
            y_train.to_csv(self.config.y_train_data_path,index=False,header=True)
            y_test.to_csv(self.config.y_test_data_path,index=False,header=True)
            X_val.to_csv(self.config.X_val_data_path,index=False,header=True)
            y_val.to_csv(self.config.y_val_data_path,index=False,header=True)

            logger.info("Ingestion of the data is completed")
            
            return(
                    self.config.X_train_data_path,
                    self.config.X_test_data_path,
                    self.config.y_train_data_path,
                    self.config.y_test_data_path,
                    self.config.X_val_data_path,
                    self.config.y_val_data_path
                )
        
        except Exception as e:
            raise CustomException(e,sys)    

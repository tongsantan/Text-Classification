import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from transformers import AutoTokenizer
from src.utils.common import save_tokenizer
from src.exception import CustomException
from src import logger
import os
import sys
from src.entity.config_entity import DataTransformationConfig

## 5. Update the components

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config=config
        
    def tokenizing(self):
        '''
        This function is responsible for tokenizing the data
        
        '''

        logger.info("Loading tokenizer from pretrained model")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        logger.info("Saving tokenizer from pretrained model")
        save_tokenizer(self.config.tokenizer_data_path, tokenizer)

        logger.info("Reading in train, val and test text data")        
        train_texts=pd.read_csv(self.config.X_train_data_path)
        train_texts = train_texts["TITLE"].to_list()
        train_labels=pd.read_csv(self.config.y_train_data_path)
        train_labels = train_labels["TARGET"].to_list()
        val_texts=pd.read_csv(self.config.X_val_data_path)
        val_texts = val_texts["TITLE"].to_list()
        val_labels=pd.read_csv(self.config.y_val_data_path)
        val_labels = val_labels["TARGET"].to_list()
        test_texts=pd.read_csv(self.config.X_test_data_path)
        test_texts = test_texts["TITLE"].to_list()
        test_labels=pd.read_csv(self.config.y_test_data_path)
        test_labels = test_labels["TARGET"].to_list()
        
        logger.info("Tokenzing the train, val and test text data") 
        train_encodings = tokenizer(train_texts, padding=True, truncation=True)
        val_encodings = tokenizer(val_texts, padding=True, truncation=True)
        test_encodings = tokenizer(test_texts, padding=True, truncation=True)

        logger.info("Generating the train, val and test datasets.") 
        batch_size = 16
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        )).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_labels
        )).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test_labels
        )).batch(batch_size)

        logger.info(f"Saving the train, val and test datasets.")
        train_dataset.save(self.config.train_dataset_path)
        val_dataset.save(self.config.val_dataset_path)
        test_dataset.save(self.config.test_dataset_path)
        
        return (
                train_dataset,
                val_dataset,
                test_dataset
            )
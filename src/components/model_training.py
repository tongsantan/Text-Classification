import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from keras.utils import to_categorical
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import os
import sys
from dataclasses import dataclass
import pickle
import warnings
warnings.filterwarnings("ignore")
from src.exception import CustomException
from src import logger
from src.entity.config_entity import ModelTrainerConfig

## 5. Update the components

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config=config

    def initiate_model_trainer(self):
        '''
        This function is responsible for model training
        
        '''
        try:
            logger.info(f"Loading the train, validation and test datasets")
            train_dataset = tf.data.Dataset.load(self.config.train_dataset_path)
            val_dataset = tf.data.Dataset.load(self.config.val_dataset_path)
            test_dataset = tf.data.Dataset.load(self.config.test_dataset_path)
            

            logger.info(f"Finetuning model starts")
            model = TFAutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",num_labels=4)
            
            num_epochs = 1

            # The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
            # by the total number of epochs. Since our dataset is already batched, we can simply take the len.
            num_train_steps = len(train_dataset) * num_epochs

            lr_scheduler = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
            )
            
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
            model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
            
            callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]

            model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, callbacks=callbacks)

            logger.info(f"Evaluating finetuned model")
            model.evaluate(test_dataset)
            
            logger.info(f"Saving finetuned model")
            model.save_pretrained(os.path.join(self.config.model_data_path))
                    
            return model
    
        except Exception as e:
            raise CustomException(e,sys)
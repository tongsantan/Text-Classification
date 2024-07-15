import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
import os
import sys
from src.utils.common import load_tokenizer
from src.exception import CustomException
from src import logger
from src.entity.config_entity import ModelEvaluationConfig


## 5. Update the components

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config=config


    def evaluate_model(self):
        '''
        This function is responsible for testing model on unseen datasets
        '''
        try:
            tokenizer = load_tokenizer(self.config.tokenizer_data_path)

            model = TFAutoModelForSequenceClassification.from_pretrained(self.config.model_data_path)
            
            text = "Pop star to start fashion company"
            logger.info("Text: {}".format(text))
            inputs = tokenizer(text, return_tensors="tf")
            output = model(inputs)
            pred_prob = tf.nn.softmax(output.logits, axis=-1)
            logger.info("{}".format(pred_prob))
            labels = ['entertainment', 'science/tech', 'business', 'health']
            logger.info("Predicted Label: {}".format(labels[np.argmax(pred_prob)]))

            text = "Revolutionary methods for discovering new materials"
            logger.info("Text: {}".format(text))
            inputs = tokenizer(text, return_tensors="tf")
            output = model(inputs)
            pred_prob = tf.nn.softmax(output.logits, axis=-1)
            logger.info("{}".format(pred_prob))
            labels = ['entertainment', 'science/tech', 'business', 'health']
            logger.info("Predicted Label: {}".format(labels[np.argmax(pred_prob)]))

            text = "Rebranded bank will target global growth"
            logger.info("Text: {}".format(text))
            inputs = tokenizer(text, return_tensors="tf")
            output = model(inputs)
            pred_prob = tf.nn.softmax(output.logits, axis=-1)
            logger.info("{}".format(pred_prob))
            labels = ['entertainment', 'science/tech', 'business', 'health']
            logger.info("Predicted Label: {}".format(labels[np.argmax(pred_prob)]))

            text = "A new sustainable vaccination against Ebola developed."
            logger.info("Text: {}".format(text))
            inputs = tokenizer(text, return_tensors="tf")
            output = model(inputs)
            pred_prob = tf.nn.softmax(output.logits, axis=-1)
            logger.info("{}".format(pred_prob))
            labels = ['entertainment', 'science/tech', 'business', 'health']
            logger.info("Predicted Label: {}".format(labels[np.argmax(pred_prob)]))

        except Exception as e:
            raise CustomException(e,sys)
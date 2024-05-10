import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customException
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def save_objects(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise customException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train,y_train)

            y_test_pred = model.predict(x_test)
            
            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customException(e,sys)

def load_objects(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_objects functions utils")
        raise customException(e,sys)
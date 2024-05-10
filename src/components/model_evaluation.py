import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score,classification_report, precision_recall_fscore_support
from src.logger.logging import logging
from src.exception.exception import customException
from src.utils.utils import load_objects

class ModelEvaluation():
    def __init__(self):
        logging.info('Evaluation Started')

    def eval_metrics(self,actual,pred):
        precision, recall, f1_score, _ = precision_recall_fscore_support(actual,pred)
        # for i,v in enumerate(['p1','p2','p3','p4']):
        #     print(f"Class {v}:")
        #     print(f"Precision : {precision[i]}")
        #     print(f"Recall : {recall[i]}")
        #     print(f"f1score: {f1_score[i]}")
        #     print()
        acc_score = accuracy_score(actual,pred)

        logging.info("evaluation metrics captured")

        return (acc_score, precision, recall, f1_score)


    def initiate_model_evaluation(self,x_train,x_test,y_train,y_test):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            model = load_objects(model_path)

            logging.info("Model has registered")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(x_test)
                accuracy , precision, recall, f1_score = self.eval_metrics(y_test,prediction)

                mlflow.log_metric("accuracy", accuracy)
                # mlflow.log_metric("precision",precision)
                # mlflow.log_metric("recall",recall)
                for i,v in enumerate(['p1','p2','p3','p4']):
                    mlflow.log_metric(f"f1-score-{v}", f1_score[i])

                if tracking_url_type_store!="file":
                    mlflow.sklearn.log_model(model,"model",registered_model_name="crm_Model")
                else:
                    mlflow.sklearn.log_model(model,"model")



        except Exception as e:
            logging.info('Error raised in Model evaluation')
            raise customException(e,sys)
        
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from src.utils.utils import save_objects,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()

    def initiate_model_training(self,df_encoded):
        try:
            y=df_encoded['Approved_Flag']
            x=df_encoded.drop(['Approved_Flag'],axis=1)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            logging.info("Splitting data frame")
            
            x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)

            
            xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

            param_grid = {
                'n_estimators':[50,100,200],
                'max_depth': [3,5,7],
                'learning_rate':[0.01,0.1,0.2]
            }

            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

            models = {
                'XGBoost' : grid_search
            }

            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)

            print("-------------------------------------------------------------")

            logging.info(f"model report: {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            print(f"Best Model is {best_model_name} with accuracy {best_model_score}")
            logging.info(f"Best Model is {best_model_name} with accuracy {best_model_score}")
            # grid_search.fit(x_train,y_train)

            # print("Best Hyperparameter : ", grid_search.best_params_)

            # logging.info(f'Best Model found with best param : {grid_search.best_params_}')

            save_objects(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )


        except Exception as e:
            logging.info("Exception occured during model training.")
            raise customException(e,sys)
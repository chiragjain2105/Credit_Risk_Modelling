import os
import sys

from src.logger.logging import logging
from src.exception.exception import customException

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

obj = DataIngestion()
raw_cs1_path, raw_cs2_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()
df_encoded = data_transformation.initialize_data_transformation(raw_cs1_path,raw_cs2_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(df_encoded)


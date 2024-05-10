import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException

import os 
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_cs1_path:str = os.path.join("artifacts","raw_cs1.xlsx")
    raw_cs2_path:str = os.path.join("artifacts","raw_cs2.xlsx")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            cs1_data = pd.read_excel('experiment\case_study1.xlsx')
            cs2_data = pd.read_excel('experiment\case_study2.xlsx')
            logging.info("reading dataframe")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_cs1_path)), exist_ok=True)
            cs1_data.to_excel(self.ingestion_config.raw_cs1_path, index=False)
            cs2_data.to_excel(self.ingestion_config.raw_cs2_path, index=False)

            logging.info("Data Ingestion Complete, I have feed the raw data-set")

            return (
                self.ingestion_config.raw_cs1_path,
                self.ingestion_config.raw_cs2_path
            )

        except Exception as e:
            raise customException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
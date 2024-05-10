import pandas as pd
import numpy as np

from src.logger.logging import logging
from src.exception.exception import customException

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self):
        pass

    def get_data_transformation(self):
        pass

    def initialize_data_transformation(self,cs1_path,cs2_path):
        try:
            cs1_df = pd.read_excel(cs1_path)
            cs2_df = pd.read_excel(cs2_path)
            logging.info("data is loaded for transformation.")

            logging.info(f'cs1 data : \n{cs1_df.head().to_string()}')
            logging.info(f'cs2 data : \n{cs2_df.head().to_string()}')

            
            # removing null values
            # cs1
            cs1_df = cs1_df.loc[cs1_df['Age_Oldest_TL']!=-99999]
            #cs2
            columns_to_be_removed = []
            for i in cs2_df.columns:
                if cs2_df.loc[cs2_df[i]==-99999].shape[0]>10000:
                    columns_to_be_removed.append(i)
            
            cs2_df = cs2_df.drop(columns=columns_to_be_removed,axis=1)

            for i in cs2_df.columns:
                cs2_df = cs2_df.loc[cs2_df[i]!=-99999]

            # merging two dataframes
            common_column = ''
            for i in list(cs1_df.columns):
                if i in list(cs2_df.columns):
                    common_column=i
            
            df = pd.merge(cs1_df,cs2_df,how='inner',left_on=[common_column], right_on=[common_column])

            logging.info(f'merged data: \n{df.head().to_string()}')

            # dropping target column
            target_var_name = 'Approved_Flag'
            # checking for categorical column
            cat_cols=[]
            for i in df.columns:
                if df[i].dtype == 'object' and i!=target_var_name:
                    cat_cols.append(i)
            # dropping categorical varible those having pval>0.05
            for col in cat_cols:
                chi2,pval,_,_=chi2_contingency(pd.crosstab(df[i],df[target_var_name]))
                if pval>0.05:
                    df = df.drop(columns=[i],axis=1)
            # finding numerical columns
            num_cols=[]
            for i in df.columns:
                if df[i].dtypes!='object' and i not in [common_column,target_var_name]:
                    num_cols.append(i)
            
            # dropping numerical coln based on Multicollinearity
            vif_data = df[num_cols]
            total_columns = vif_data.shape[1]
            columns_to_be_kept = []
            column_index = 0

            for i in range(0,total_columns):

                vif_value = variance_inflation_factor(vif_data,column_index)

                if vif_value<=6:
                    columns_to_be_kept.append(num_cols[i])
                    column_index+=1
                else:
                    vif_data = vif_data.drop([num_cols[i]],axis=1)
            
            # anova test to drop columns
            columns_to_be_kept_numerical = []

            for i in columns_to_be_kept:
                a = list(df[i])
                b = list(df[target_var_name])

                group_p1 = [value for value,group in zip(a,b) if group=='P1']
                group_p2 = [value for value,group in zip(a,b) if group=='P2']
                group_p3 = [value for value,group in zip(a,b) if group=='P3']
                group_p4 = [value for value,group in zip(a,b) if group=='P4']

                f_statistics,p_value = f_oneway(group_p1,group_p2,group_p3,group_p4)

                if p_value<=0.05:
                    columns_to_be_kept_numerical.append(i)
            
            features = columns_to_be_kept_numerical+cat_cols
            df = df[features+[target_var_name]]

            # treating ordinal column - EDUCATION
            df.loc[df['EDUCATION']=='SSC',['EDUCATION']]=1
            df.loc[df['EDUCATION']=='GRADUATE',['EDUCATION']]=3
            df.loc[df['EDUCATION']=='12TH',['EDUCATION']]=2
            df.loc[df['EDUCATION']=='UNDER GRADUATE',['EDUCATION']]=3
            df.loc[df['EDUCATION']=='POST-GRADUATE',['EDUCATION']]=4
            df.loc[df['EDUCATION']=='OTHERS',['EDUCATION']]=1
            df.loc[df['EDUCATION']=='PROFESSIONAL',['EDUCATION']]=3

            df['EDUCATION'] = df['EDUCATION'].astype(int)

            # one hot encoding
            df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'],dtype='int')

            logging.info(f'prepared data: \n {df_encoded.head()}')

            logging.info('Done with data transformation')

            logging.info("Preparing train and test data")

            y=df_encoded['Approved_Flag']
            x=df_encoded.drop(['Approved_Flag'],axis=1)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            logging.info("Splitting data frame")
            
            x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)

            return (x_train,x_test,y_train,y_test)





        except Exception as e:
            logging.info('Exception occured during data transformation.')
            raise customException(e,sys)
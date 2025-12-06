import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    Preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation.

        '''
        try:
            num_columns = ["writing_score","reading_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )


            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=True)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns standard scaling  completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)

                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data ompleted")

            logging.info("Obtaining preprocessing objects")

            preprocessing_obj = self.get_data_transformer_obj()

            target_col = 'math_score'
            num_columns = ["writing_score","reading_score"]

            input_train_feature = train_df.drop(columns=[target_col],axis=1)
            target_train_feature = train_df[target_col]

            input_test_feature = test_df.drop(columns=[target_col], axis=1)
            target_test_feature = test_df[target_col]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")


            input_train_feature_arr = preprocessing_obj.fit_transform(input_train_feature)
            input_test_feature_arr = preprocessing_obj.transform(input_test_feature)

            train_arr = np.c_[input_train_feature_arr, np.array(target_train_feature)]
            test_arr = np.c_[input_test_feature_arr, np.array(target_test_feature)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.Preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.Preprocessor_obj_file_path
            )
           
        except Exception as e:
            raise CustomException(e,sys)
            









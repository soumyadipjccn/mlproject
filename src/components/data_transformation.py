import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

# This class defines a data structure using the dataclass decorator from Python's dataclasses module.

# It holds configuration settings for data transformation, such as the file path for saving preprocessing objects.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    # method initializes the class instance and sets up the data transformation configuration.
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

         # Method to obtain a preprocessing object for data transformation
        # It initializes pipelines for numerical and categorical features, applies imputation and scaling.
        # Returns a ColumnTransformer object combining both pipelines.
        try:
            logging.info("Data Transformation initiated")

            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Columns Standard Scaling Completed")
            logging.info("Categorical Columns encoding Completed")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in the Inittiate_data_transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        # Method to initiate data transformation
        # Reads train and test data from CSV files
        # Applies preprocessing object obtained from get_data_transformer_object() on both train and test data
        # Saves preprocessing object to a file
        # Returns transformed train and test data arrays along with the file path of the saved preprocessing object
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)
        

# Both methods in the DataTransformation class have exception handling to catch and log any errors that may occur during data transformation. If an error occurs, it raises a CustomException with the error details.

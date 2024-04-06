# for operating system related functionality.
import os
# for accessing Python interpreter settings.
import sys

#  a custom exception class from src.exception.
from src.exception import CustomException


# or logging messages.
from src.logger import logging

# for data manipulation and analysis.
import pandas as pd
#  for splitting data into training and testing sets.
from sklearn.model_selection import train_test_split

# for creating data classes.
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer





# A data class DataIngestionConfig is defined using @dataclass decorator.
@dataclass
class DataIngestionConfig:

    # It contains attributes train_data_path, test_data_path, and raw_data_path, initialized with default values using os.path.join().
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


# A class named DataIngestion is defined.
class DataIngestion:
    
    # The __init__() method initializes an instance of the class with an attribute ingestion_config which is an instance of DataIngestionConfig.
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

     # method is defined It logs a message indicating the entry into the data ingestion method.:

    def initiate_data_ingestion(self):
        logging.info("Entered The Data Ingestion method or component")

        try:

            #Tries to read a CSV file named "stud.csv" located in the "notebook\data" directory into a pandas DataFrame (df).
            df = pd.read_csv("notebook\\data\\stud.csv")

            logging.info("Read The Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            # Saves the training and testing sets as CSV files.
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Part Completed")

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        # Catches any exceptions raised during execution, logs the exception, and raises a CustomException.
        except Exception as e:
            logging.info()
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj = DataIngestion()

    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)[:2]


    modeltrainer = ModelTrainer()
    print(ModelTrainer().initate_model_training(train_arr, test_arr))



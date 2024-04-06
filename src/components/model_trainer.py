
# These are the necessary imports for the script. It imports modules and classes required for model training and evaluation, logging, handling exceptions, and utilities for saving objects and evaluating models.
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models



#  Here, a data class ModelTrainerConfig is defined using the dataclass decorator from the dataclasses module. It holds the configuration settings for the ModelTrainer class, particularly the file path where the trained model will be saved.
@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')




# The ModelTrainer class is defined. It has an __init__ method where an instance of ModelTrainerConfig is created, holding the configuration settings for the model trainer.
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()



    # This method takes train_array and test_array as input, presumably containing training and testing data respectively.

    def initate_model_training(self, train_array, test_array):



        # The method starts by splitting the dependent and independent variables from the provided training and testing data arrays. It's assumed that the last column in each array is the dependent variable (target) and the preceding columns are independent variables (features).
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )


            # A dictionary models is created where keys are model names and values are instances of various regression models initialized using their respective constructors.

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRFRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=True),
                "Adaboost Classifier": AdaBoostRegressor(),
            }



            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }



            # The evaluate_model() function is called with training and testing data along with the dictionary of models. It returns a dictionary containing evaluation metrics for each model.

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')


            # The model evaluation report is logged, and the best model is selected based on the maximum value of evaluation metric (R2 score in this case).

            # Get best model
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            

            # The best model is saved to a file using the save_object() utility function.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report[best_model_name]

        except Exception as e:
            logging.error('Exception occurred at Model Training')
            raise CustomException(e, sys)
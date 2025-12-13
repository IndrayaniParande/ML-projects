import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostingRegressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            params = {
                "Random Forest":{
                    "n_estimators":[8,16,32,64,128,256]
                },
                "Decision Tree" :{
                    "criterion":['squared_error', 'friedman_mse','poisson'],
                    "splitter":['best', 'random']
                },
                "Gradient Boosting":{
                    "loss":['squared_error', 'absolute_error', 'huber', 'quantile'],
                    "learning_rate":[.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                    "n_estimators":[8,16,32,64,128,256]
                },
                "Linear Regression":{}
                ,
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    "weights":['uniform','distance'],
                    "algorithm":['auto','ball_tree','kd_tree','brute']   
                },
                "XGBRegressor":{
                    "learning_rate":[.1,.01,.05,.001],
                    "n_estimators":[8,16,32,64,128,256]

                },
                "CatBoostingRegressor":{
                    "depth":[6,8,10],
                    "learning_rate":[0.1,0.05,0.01],
                    "iterations":[30,50,100]
                },
                "AdaBoost Regressor":{
                    "learning_rate":[.1,.01,0.5,.001],
                    "n_estimators":[8,16,32,64,128,25]
                }

            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score = max(sorted(model_report.values()))
            best_mdoel_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_mdoel_name]

            if best_model_score<0.6:
                raise CustomException("No best model found!")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_Score = r2_score(y_test, predicted)
            return r2_Score



        except Exception as e:
            raise CustomException(e,sys)

        
        

from sklearn.linear_model import LinearRegression , Ridge, Lasso
from sklearn.svm import LinearSVR , SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor , BaggingRegressor , GradientBoostingRegressor , RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import json
import pandas as pd
from logging_code import logger
import pickle

models = {
    "Linear Regression" : LinearRegression() ,
    "Ridge Regression"  : Ridge() ,
    "Lasso Regression"  : Lasso() ,
    "Linear Support Vector Regression" : LinearSVR() ,
    "Decision Tree Regression" : DecisionTreeRegressor() ,
    "AdaBoost Regression" : AdaBoostRegressor() ,
    "Bagging Regression" : BaggingRegressor() ,
    "Gradient Boosting Regression" : GradientBoostingRegressor() ,
    "Random Forest Regression" : RandomForestRegressor() ,
    "KNN Regression" : KNeighborsRegressor() ,
    "Support Vector Regression" : SVR()
}

parameters = {
    "Gradient Boosting Regression" : {"learning_rate" : [0.1 , 0.01 , 0.001] , "n_estimators" : [50 , 100 , 150 , 200] , "criterion" : ["friedman_mse" ,  "squared_error"]} ,
    "AdaBoost Regression" : {"n_estimators" : [50 , 100 , 150 , 200] ,"learning_rate" : [0.1 , 0.01 , 0.001] , "loss" : ["linear", "square", "exponential"]} ,
    "Bagging Regression" : {"n_estimators" : [5 , 10 , 15 , 20] , "max_features" : [1 , 2 , 3]} ,
    "Random Forest Regression" : {"n_estimators" : [50 , 100 , 150 , 200] , "criterion" : ["squared_error", "absolute_error", "friedman_mse", "poisson"]} ,
    "Linear Support Vector Regression" : {"loss" : ["epsilon_insensitive", "squared_epsilon_insensitive"]} ,
    "KNN Regression" : {"n_neighbors" : [3 , 5 , 7],"weights" : ["uniform", "distance"]} ,
    "Decision Tree Regression" : {"max_depth" : [5 , 7 , 10] , "criterion" : ["squared_error" , "friedman_mse" , "absolute_error"] , "splitter" : ["best", "random"]} ,
    "Ridge Regression" : {"alpha" : [0.5 , 1 , 1.5]} ,
    "Support Vector Regression" : {"kernel" : ["linear", "poly", "rbf", "sigmoid"]} ,
    "Lasso Regression" : {"alpha" : [0.5 , 1 , 1.5]}
}

# Store trained models separately
trained_models = {}
r2_scores = {}  # Fixed variable name

def training_models(train_data_features, labels):
    train_models_summary = {}
    logger.info("Started training_models function")
    for model_name, model in models.items():
        if model_name in parameters:
            cv = RandomizedSearchCV(estimator=model, param_distributions=parameters[model_name], n_iter=10, random_state=42)
            cv.fit(train_data_features, labels)
            # Update the model with the best estimator
            trained_models[model_name] = cv.best_estimator_
            train_models_summary[model_name] = {
                "accuracy": cv.best_score_ * 100,
                "best_params": cv.best_params_
            }
            logger.info(f"Trained {model_name} model with hyperparameter tuning")
        else:
            model.fit(train_data_features, labels)
            trained_models[model_name] = model
            accuracy = model.score(train_data_features, labels)
            train_models_summary[model_name] = accuracy
            logger.info(f"Trained {model_name} model")
    
    # Save only serializable data to JSON
    with open("training_model_accuracy.json", "w") as f:
        json.dump(train_models_summary, f, indent=4)
    
    logger.info("Completed training all models")
    

def testing_models(test_data_features, labels):
    test_model_summary = {}
    global r2_scores
    logger.info("Started testing_models function")
    for model_name, model in trained_models.items():  # Use trained_models
        accuracy = model.score(test_data_features, labels)
        logger.info(f"Tested {model_name} model")
        test_model_summary[model_name] = accuracy
        
        # Calculate R2 score
        predictions = model.predict(test_data_features)
        r2_scores[model_name] = r2_score(labels, predictions)  # Use function, not variable
    
    logger.info("Completed testing all models")
    logger.info(f"Testing model summary: {test_model_summary}")

    with open("testing_model_accuracy.json", "w") as f:
        json.dump(test_model_summary, f, indent=4)
    with open("testing_model_r2_score.json", "w") as f:
        json.dump(r2_scores, f, indent=4)
    logger.info("Saved testing model accuracies and R2 scores to JSON files")   

def save_best_model():
    logger.info("Started save_best_model function")
    with open("testing_model_r2_score.json", "r") as f:
        model_accuracies = json.load(f)
    
    best_model_name, max_accuracy = max(model_accuracies.items(), key=lambda item: item[1])
    
    # Save model info
    with open("best_model.json", "w") as f:
        json.dump({"Best Model": best_model_name, "Accuracy": max_accuracy}, f, indent=4)
    
    logger.info(f"Best Model: {best_model_name} with accuracy: {max_accuracy}")

    # Save the actual trained model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(trained_models[best_model_name], f)
    
    logger.info(f"Saved the best model: {best_model_name}")

if __name__ == "__main__":
    training_data_features = pd.read_csv("Preprocessed_Data/X_train.csv")
    train_labels = pd.read_csv("Preprocessed_Data/y_train.csv")
    testing_data_features = pd.read_csv("Preprocessed_Data/X_test.csv")
    test_labels = pd.read_csv("Preprocessed_Data/y_test.csv")
    
    training_models(training_data_features, train_labels)
    testing_models(testing_data_features, test_labels)
    save_best_model()
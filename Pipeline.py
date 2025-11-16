from data_collection import download_dataset
from data_loading import train_test_split_data
from data_transformation import preprocess_data
from model_training import training_models, testing_models, save_best_model
import pandas as pd
import schedule
import time
from logging_code import logger




def ml_pipeline():
    logger.info("Started ML pipeline")
    kaggle_dataset = "mosapabdelghany/medical-insurance-cost-dataset"
    download_dataset(kaggle_dataset)
    train_test_split_data("Datasets/insurance.csv")
    preprocess_data("Datasets/train.csv", target_column="charges", name="train")
    preprocess_data("Datasets/test.csv", target_column="charges", name="test")
    training_data_features = pd.read_csv("Preprocessed_Data/X_train.csv")
    train_labels = pd.read_csv("Preprocessed_Data/y_train.csv") 
    testing_data_features = pd.read_csv("Preprocessed_Data/X_test.csv")
    test_labels = pd.read_csv("Preprocessed_Data/y_test.csv")
    training_models(training_data_features, train_labels)
    testing_models(testing_data_features, test_labels)
    save_best_model()
    logger.info("Completed ML pipeline")


if __name__ == "__main__":
    
    ml_pipeline()

    
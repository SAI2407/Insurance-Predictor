from sklearn.model_selection import train_test_split
import pandas as pd
from logging_code import logger
import os


def train_test_split_data(dataset_path:str , test_size : float = 0.25 , random_state : int = 42 , shuffle : bool = True , stratify = None) -> None:
    """ This function splits the dataset into training and testing sets."""
    logger.info("Started train_test_split_data function")
    df = pd.read_csv(dataset_path , index_col=0)
    logger.info("Loaded data")
    train_data , test_data = train_test_split(df , test_size = test_size , random_state = random_state , shuffle = shuffle , stratify = stratify)
    logger.info("Split data")
    os.makedirs("Datasets" , exist_ok = True)
    train_data.to_csv(os.path.join("Datasets" , "train.csv"))
    test_data.to_csv(os.path.join("Datasets" , "test.csv"))
    logger.info("Saved data")
    logger.info("Completed train_test_split_data function")
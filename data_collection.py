import kaggle
from logging_code import logger


def download_dataset(kaggle_dataset:str) -> None:
    kaggle.api.authenticate()
    logger.info("Authenticated with Kaggle API")
    kaggle.api.dataset_download_files(kaggle_dataset, path='Datasets/', unzip=True)
    logger.info("Downloaded and extracted dataset from Kaggle")
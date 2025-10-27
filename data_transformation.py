import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from logging_code import logger
import pickle


 


def preprocess_data(data_path : str , target_column : str , name : str) -> None:
    """ This function preprocesses the training data."""
    logger.info("Started preprocess_data function")
    df = pd.read_csv(data_path)
    logger.info("Loaded data")
    

    # Splitting the features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]
    logger.info("Split features and target variable")

    features = list(X.columns)

    # Identifying numerical and categorical features
    numerical_features = [feature for feature in features if df[feature].dtype == 'int64' or df[feature].dtype == 'float64']
    categorical_features = [feature for feature in features if df[feature].dtype == 'object']

    # Handling missing values
    df[categorical_features].fillna("Unknown", inplace=True)
    df[numerical_features].fillna(df[numerical_features].mean(), inplace=True)
    logger.info("Handled missing values")

    

    # Encoding categorical features
    logger.info("Started encoding categorical features")
    enc = OneHotEncoder(feature_name_combiner = "concat")
    enc.fit(X[categorical_features])
    X_encoded = enc.transform(X[categorical_features]).toarray()
    numeric_categorical_features = list(enc.get_feature_names_out(categorical_features))
    X[numeric_categorical_features] = X_encoded
    X.drop(categorical_features , axis = 1 , inplace = True)
    logger.info("Encoded categorical features")

    # Scaling numerical features
    logger.info("Started scaling numerical features")
    scaler = StandardScaler()
    scaler.fit(X[numerical_features])
    X[numerical_features] = scaler.transform(X[numerical_features])
    logger.info("Scaled numerical features")

    # Save the encoder and scaler
    os.makedirs("Transformers", exist_ok=True)
    with open(os.path.join("Transformers", "encoder.pkl"), "wb") as f:
        pickle.dump(enc, f)
    with open(os.path.join("Transformers", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved encoder and scaler")

    # Saving the preprocessed data
    os.makedirs("Preprocessed_Data" , exist_ok = True)
    X.to_csv(os.path.join("Preprocessed_Data" , f"X_{name}.csv") , index = False)
    y.to_csv(os.path.join("Preprocessed_Data" , f"y_{name}.csv") , index = False)
    logger.info("Saved preprocessed training data")
    logger.info("Completed preprocess_data function")





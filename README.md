# Insurance Cost Predictor

A production-ready web application that predicts insurance costs based on user inputs such as age, height, and other personal details.  
The backend is built with **FastAPI**, containerized using **Docker**, and deployed on **AWS EC2**, with container images stored securely in **AWS ECR**.

## Features

### 🔹 Insurance Cost Prediction
- Predicts insurance cost using a trained machine learning model.

### 🔹 Automated ML Training Pipeline
- Downloads raw data from **Kaggle**.
- Performs **data preprocessing**, cleaning, feature engineering, and transformation.
- Splits the data into **training** and **testing** sets.
- Trains multiple machine learning models using **cross-validation**.
- Automatically selects and saves the **best-performing model** based on evaluation metrics (e.g., accuracy).

### 🔹 Integrated Logging System
Logs are generated for tracking:
- Data pipeline execution  
- Model training workflows  
- Error handling and debugging  

### 🔹 Dockerized Deployment
- Dockerized for consistent and reproducible deployment across environments.

### 🔹 CI/CD Pipeline (GitHub Actions)
The CI/CD workflow automatically:
- Builds the Docker image  
- Runs tests  
- Pushes the image to **AWS ECR**  
- Deploys the latest version automatically to **AWS EC2**

### 🔹 Cloud Deployment
- Hosted on **AWS EC2**  
- Accessible securely through **port 80**

## Web Interface

Below is the web interface of the Insurance Predictor:

![Insurance Predictor UI](https://github.com/user-attachments/assets/33f887bf-2347-4123-a7b1-f6570940b3fe)




from fastapi import FastAPI
from pydantic import Field , computed_field , BaseModel
from typing import Annotated
import pandas as pd 
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from logging_code import logger
import numpy as np

from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi import  Form
import time
import os

LOG_FILE = "user_predictions.csv"
base_path = os.path.dirname(os.path.abspath(__file__))



if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "age", "bmi", "children", "sex", "smoker", "region",
        "predicted_cost", "latency"
    ]).to_csv(LOG_FILE, index=False)

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def load_model(model_path):
    model_path = os.path.join(base_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model




def load_transformers(encoder_path, scaler_path):
    encoder_path = os.path.join(base_path, encoder_path)
    scaler_path = os.path.join(base_path, scaler_path)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return encoder, scaler

def preprocess_data(df):
    encoder, scaler = load_transformers("Transformers/encoder.pkl", "Transformers/scaler.pkl")
    cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    df = df.reindex(columns=cols)
    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Numerical features: {numerical_features}")
    logger.info(f"Categorical features: {categorical_features}")
    
    # Transform
    df_encoded = encoder.transform(df[categorical_features]).toarray()
    df_scaled = scaler.transform(df[numerical_features])

    # Build DataFrames with correct column names
    encoded_cols = list(encoder.get_feature_names_out(categorical_features))
    df_encoded = pd.DataFrame(df_encoded, columns=encoded_cols)
    df_scaled = pd.DataFrame(df_scaled, columns=numerical_features, index=df.index)
    logger.info("Transformed data")

    # Combine back
    df_final = pd.concat([df_scaled, df_encoded], axis=1)
    logger.info(f"Preprocessed data: {df_final.head()}")

    return df_final


class InsurancePerson(BaseModel):
    age : Annotated[int , Field(..., description="Enter the age" , examples= [55])]
    sex : Annotated[str , Field(... , description="Gender of the person either male or female" , examples = ["male"])]
    height : Annotated[float , Field(... , description="Enter the height in meters" , examples=[1.72])]
    weight : Annotated[float , Field(...,description="Weight of the person in Kgs" , examples=[45.0])]
    children : Annotated[int , Field(..., description="Number of children" , examples=[2])]
    smoker : Annotated[str , Field(..., description="Whether the person is a smoker or not" , examples=["yes"])]
    region : Annotated[str , Field(..., description="Region of the person choose from [southeast , southwest , northwest ]" , examples=["southwest"])]


    @computed_field(return_type=float)
    @property
    def bmi(self)-> float:
        return self.weight / self.height ** 2
    


def run_prediction(person: InsurancePerson) -> dict:
    try:
        df = pd.DataFrame([person.model_dump()])
        logger.info(f"Input data: {df.head()}")

        df = df.drop(columns=['height', 'weight'], axis=1)
        logger.info(f"Data after dropping height and weight: {df.head()}")

        df_preprocessed = preprocess_data(df)
        logger.info(f"Data after preprocessing: {df_preprocessed.head()}")

        model = load_model("best_model.pkl")

        input_aligned = df_preprocessed.reindex(columns=model.feature_names_in_, fill_value=0)
        logger.info(f"Data after aligning columns: {input_aligned.head()}")

        prediction = model.predict(input_aligned)

        return {
            "predicted_insurance_cost": round(prediction[0], 2),
            "bmi": person.bmi
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}
    

@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form")
async def predict_form(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...)
):
    try:
        person_data = InsurancePerson(
            age=age,
            sex=sex,
            height=height,
            weight=weight,
            children=children,
            smoker=smoker,
            region=region
        )

        start_time = time.time()
        prediction_result = run_prediction(person_data)
        end_time = time.time()
        latency = end_time - start_time
        new_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "age": person_data.age,
        "bmi": round(prediction_result.get("bmi", 0), 2),
        "children": person_data.children,
        "sex": person_data.sex,
        "smoker": person_data.smoker,
        "region": person_data.region,
        "predicted_cost": prediction_result["predicted_insurance_cost"],
        "latency": latency
        }

        pd.DataFrame([new_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)



        return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "age": person_data.age,
            "bmi": round(prediction_result.get("bmi", 0), 2),
            "cost": prediction_result.get("predicted_insurance_cost", 0),
            "latency": round(latency, 3)  # round to 3 decimal places
        }
        )


    except Exception as e:
        logger.error(f"Form processing error: {str(e)}")
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": str(e)}
        )

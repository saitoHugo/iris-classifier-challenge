from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from pathlib import Path
from utils import utils
from typing import List
app = FastAPI()


"""
TODO: 
    - load all models gloabally
    - create a train route
    - create a predict route based on model selection
    - extra route get_all_trained_models
    - download local data or download from bucket

    - add logging
"""

BASE_DIR = Path(__file__).resolve(strict=True).parent

#load datset globally

#load all model glabally if exists


### pydantic models
class Input(BaseModel):
    model_name: str
    Inputs: List[float]



class Prediction(BaseModel):
    label: str
    value: int


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

@app.post("/predict", response_model=Prediction, status_code=200)
def get_prediction(payload: Input):
    model_name = payload.model_name

    prediction_list = utils.predict(ticker)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {"inputs": ticker, "prediction": prediction_list}
    return response_object
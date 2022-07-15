from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from pathlib import Path
from utils import utils
from typing import List
import logging
from utils import global_varibles
logger = logging.getLogger(__name__)

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
# #TODO: define global base dir
# global BASE_DIR
# BASE_DIR = Path(__file__).resolve(strict=True).parent

#load datset globally

#load all model glabally if exists


### pydantic models
class PredictionInput(BaseModel):
    model_name: str
    Inputs: List[float]



class PredictionOutput(BaseModel):
    model_name: str
    label: str
    value: int

class TrainInput(BaseModel):
    #TODO:update attributes
    model_name: str
    Inputs: List[float]



class TrainOutput(BaseModel):
    #TODO:update attributes
    model_name: str
    label: str
    value: int


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

@app.post("/predict", response_model=PredictionOutput, status_code=200)
def get_prediction(payload: PredictionInput):
    logger.info("/predict route init")
    model_name = payload.model_name

    prediction_list = utils.predict(model_name)
    logger.info("utils.predict executed")
    

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")
    
    value = prediction_list[0]

    #convert to label
    label = utils.convert_value_to_label(value)
    logger.info("utils.convert_value_to_label executed")
    
    response_object = {"model_name": model_name, "label":label, "value": value}
    return response_object


@app.post("/train", response_model=TrainOutput, status_code=200)
def get_prediction(payload: TrainInput):
    logger.info("/train route init")
    model_name = payload.model_name

    prediction_list = utils.train(model_name)
    logger.info("utils.predict executed")
    

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")
    
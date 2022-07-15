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

# global BASE_DIR
# BASE_DIR = Path(__file__).resolve(strict=True).parent


"""
TODO: 
    - extra route get_all_trained_models
    - download local data or download from bucket

"""



### pydantic models
class PredictionInput(BaseModel):
    model_name: str
    inputs: List[float]

class PredictionOutput(BaseModel):
    model_name: str
    label: str
    value: int

class TrainOutput(BaseModel):
    LogisticRegression : List[float]
    SupportVectorMachine : List[float]
    KNearestNeighbor : List[float]
    RandomForest : List[float]
    XgBoost : List[float]
    GaussianNaiveBayes: List[float]




@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post("/train", response_model=TrainOutput, status_code=200)
def train():
    logger.info("/train route init")
    try:
        results = utils.execute_train()
        logger.info("utils.execute_train executed")
    except:
        logger.info("error occurred in the training process")
        raise HTTPException(status_code=500, detail="Internal error during training execution.")

    return results
    

@app.post("/predict", response_model=PredictionOutput, status_code=200)
def prediction(payload: PredictionInput):
    logger.info("/predict route init")
      
    if not (payload.model_name or payload.inputs):
        logger.info("Model Name was not send via payload")
        raise HTTPException(status_code=400, detail="Payload error: model_name attribute could not be found")
    model_name = payload.model_name
    print(f"Model Name - > {model_name}")
    inputs = payload.inputs 
    print(f"Model inputs - > {inputs}")

    

    #execute prediction
    try:
        prediction = utils.predict(model_name, inputs)
        print(f"Prediction -> {prediction}")
        print(f"Prediction type-> {type(prediction)}")
        logger.info("utils.predict executed")
    except:
        logger.info("error occurred in the prediction process")
        raise HTTPException(status_code=500, detail="Internal error during predict execution.")
    

    if not (prediction or len(prediction) == 1):
        raise HTTPException(status_code=500, detail="Internal Problem with predict execution.")
    
    
    value = prediction[0]
    print(f"Prediction value - > {value}")

    #convert to label
    label = utils.convert_value_to_label(value)
    print(f"Prediction label - > {label}")
    logger.info("utils.convert_value_to_label executed")
    
    response_object = {"model_name": model_name, "label":label, "value": value}
    print(f"Prediction response_object - > {response_object}")
    return response_object
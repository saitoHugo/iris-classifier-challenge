from pydantic import BaseModel
from typing import List

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



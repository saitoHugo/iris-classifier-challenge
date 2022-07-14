from typing import List

def train():
    #download dataset

    #load dataset globally

    #execute preprocessing

    #execute train

    #save new models

    #return 200

    pass

def predict(model_name:str, input:List[float]):
    model_path = Path(BASE_DIR).joinpath(f"{ticker}.joblib")
    if not model_path.exists():
        return Exception(f"Could not find model at {model_path}")
    pass
    ### VALIDATE INPUT

    #load model base on given name

    #execute prediction

    #return output class by name
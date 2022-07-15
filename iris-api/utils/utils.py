from typing import List
from pathlib import Path
import global_varibles

def train():
    #download dataset

    #load dataset globally

    #execute preprocessing

    #execute train

    #save new models

    #return 200

    pass

def predict(model_name:str, input:List[float]) -> str:
    model_path = Path(global_varibles.BASE_DIR).joinpath(f"{model_name}.pkl")
    if not model_path.exists():
        return Exception(f"Could not find model at {model_path}")
    pass
    ### VALIDATE INPUT

    #load model base on given name

    #execute prediction

    #return output class by name (str)

def convert_value_to_label(prediction_value:int):
    label_decode_dict = {
        0 : 'flower0',
        1 : 'flower1',
        2 : 'flower2',
    }

    if prediction_value not in label_decode_dict.keys():
        raise Exception(f"predict value given {prediction_value} does not exists in decode dict. Options are 0, 1 or 2 (int).")
    return label_decode_dict[prediction_value]
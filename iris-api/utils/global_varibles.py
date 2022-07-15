from pathlib import Path
import pandas as pd
"""
File to save all global variables declared in api
"""
#define root path
global BASE_DIR
BASE_DIR = Path(__file__).resolve(strict=True).parent


#load dataset globally
column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species' ]
global data
data = pd.read_csv('../data/iris-data.csv', names=column_names, header=None)

#load all model glabally if exists
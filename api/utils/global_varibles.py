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
#load each model as global varible
# classifiers_name = {
#         'Logistic Regression':'log_reg',
#         'Support Vector Machine':'svm',
#         'K Nearest Neighbor': 'knn',
#         'Random Forest':'rand_forest',
#         'Xg Boost':'xgb',
#         'Gaussian Naive Bayes':'gaussian',
#     }

# models_dir = '../models/prod/'
# for clf in classifiers_name.keys():
#     model_name = clf.replace(' ', '') + '.pkl'
#     model_path = models_dir + model_name
    


#load decode dict label
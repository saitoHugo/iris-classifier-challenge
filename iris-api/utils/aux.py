from typing import List
from pathlib import Path
#import api.utils.globals as globals
import utils.globals as globals
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json 
from datetime import datetime
import numpy as np
import os

#Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import logging
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


##################################
############ TRAINING ############
##################################

def execute_train():
    #TODO: download dataset if not erxists

    #load dataset globally
    #['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
    column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species' ]
    #print(f"column_names -> {column_names}")
    #DONE: update to BASE_DIR
    # print(f"BASE_DIR -> {globals.BASE_DIR}")
    # print(f"BASE_DIR -> {type(globals.BASE_DIR)}")
    # #print(Path(__file__).resolve(strict=True).parent.parent.parent)

    # print(f"BASE_DIR.parent.parent -> {globals.BASE_DIR.parent.parent}")
    # print(f"BASE_DIR .parent.parent type -> {type(globals.BASE_DIR.parent.parent)}")

    #project_root = Path(__file__).resolve(strict=True).parent.parent.parent
    #print(f"project_root -> {project_root}")
    #data_path = os.path.join(project_root, "/data/iris-data.csv")
    data_path = globals.BASE_DIR.parent.parent / "data/iris-data.csv"
    # print(f"data_path -> {data_path}")
    # print(f"data_path type -> {type(data_path)}")
    
    data = pd.read_csv(data_path, names=column_names, header=None)
    
    #data = pd.read_csv('../../../data/iris-data.csv', names=column_names, header=None)
    #print(f"data -> {data}")
    #execute preprocessing
    x_train, x_test, y_train, y_test = data_pipeline(data=data)

    #execute train with all models
    trained_classifiers = train_all_models(x_train, y_train)
    
    #execute test
    #save json result
    results = performance_analysis(trained_classifiers, x_train, x_test, y_train, y_test)
    

    #save new models
    save_trained_models(trained_classifiers)

    #return outputs
    return results


def convert_value_to_label(prediction_value:int) ->  str:
    #DONE: update to correct label names
    label_decode_dict = {
        0 : 'Iris-setosa',
        1 : 'Iris-versicolor',
        2 : 'Iris-virginica',
    }

    if prediction_value not in label_decode_dict.keys():
        raise Exception(f"predict value given {prediction_value} does not exists in decode dict. Options are 0, 1 or 2 (int).")
    return label_decode_dict[prediction_value]

def data_pipeline(data:pd.DataFrame) -> pd.DataFrame:
    """
    Execute data preprocessing on raw data
    parms: data - pandas.Dataframe
    output: preprocessed_data - 
    """
    logger.info("data_pipeline init")
    #Separate feature from label
    data_x = data.drop(['Species'], axis=1)
    print(f"data_x shape - {data_x.shape}")
    print(f"data_x columns - {data_x.columns}")
    print(f"data_x head - {data_x.head()}")

    data_y = data['Species']
    #print(f"data_y type - {type(data_y)}")
    #print(f"data_y shape - {data_y.shape}")
    #print(f"data_y head - {data_y.head()}")
    #print(f"data_y head - {data_y.tail()}")

    #convert Labels to numerical
    label_encoder = LabelEncoder()
    data_y = label_encoder.fit_transform(data_y)
    #print(f"data_y type - {type(data_y)}")
    #print(f"data_y head - {data_y[:10]}")
    #print(f"data_y tail - {data_y[-10:]}")
    #print(f"data_y lenght - {len(data_y)}")
    logger.info("data transformations executed")


    x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.15,random_state=0)
    logger.info("train_test_split executed")
    print(f"x_train type - {type(x_train)}")
    print(f"y_train type - {type(y_train)}")
    return x_train,x_test,y_train,y_test

def train_all_models(x_train, y_train):
    logger.info("train_all_models init")
    #Initialize all models objects
    classifiers = define_all_classifiers()
    for clf in classifiers.keys(): 
        #Execute train
        classifiers[clf].fit(x_train,y_train)
        logger.info(f"Model {clf} TRAINED!")
    logger.info("train_all_models end")
    return classifiers


def define_all_classifiers():
    #LogisticRegression
    log_reg = LogisticRegression(max_iter=150,multi_class='multinomial')

    #SVM
    svm = SVC()

    #KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5)

    #RandomForestClassifier
    rand_forest = RandomForestClassifier(n_estimators=150)

    #XGBClassifier
    xgb = XGBClassifier(n_estimators=80,learning_rate=0.01)

    #GaussianNB
    gaussian = GaussianNB()


    classifiers = {
        'Logistic Regression':log_reg,
        'Support Vector Machine':svm,
        'K Nearest Neighbor': knn,
        'Random Forest':rand_forest,
        'Xg Boost':xgb,
        'Gaussian Naive Bayes':gaussian,
    }
    return classifiers

def performance_analysis(trained_classifiers, x_train, x_test, y_train, y_test):
    results = {}
    for clf in trained_classifiers.keys():
        #Test on testing dataset
        y_pred = trained_classifiers[clf].predict(x_test)
        
        #Performance Metrics
        train_score = round(trained_classifiers[clf].score(x_train, y_train)*100, 4)
        
        acc_score = round(accuracy_score(y_test,y_pred)*100, 4)
        precision = round(precision_score(y_test, y_pred,average='micro')*100, 4)
        conf_matrix = confusion_matrix(y_test, y_pred,)
        recall =  round(recall_score(y_test, y_pred,average='micro')*100, 4)
        f1 = round(f1_score(y_test,y_pred,average='micro')*100, 4)
        
        #DONE: update to save into json
        #print(f"{clf} ------->> train score = {train_score}%")
        #print(f"{clf} ------->> accuracy_score = {acc_score}%")
        #print(f"{clf} ------->> precision = {precision}%")
        #print(f"{clf} ------->> conf_matrix = ")
        #print(f"{conf_matrix}")
        #print(f"{clf} ------->> recall = {recall}%")
        #print(f"{clf} ------->> f1 score = {f1}%")
        results[clf.replace(' ', '')] = [train_score, acc_score, precision, recall, f1]
        logger.info(f"Model {clf} EVALUATED!")
    if results:
        #save new results
        #DONE: add relative path
        # results_path = globals.BASE_DIR.parent.parent / "results/"
        # print(f"results_path -> {results_path}")
        # print(f"results_path type -> {type(results_path)}")
    
        #results_path = '../results/'
        timestamp = datetime.now().strftime("%d-%B-%Y") + '-' + datetime.now().time().strftime("%H-%M-%S")
        filename ='results/results-'+timestamp+'.json'
        file_path = globals.BASE_DIR.parent.parent / filename
        print(f"file_path -> {file_path}")
        print(f"file_path type -> {type(file_path)}")
    
        with open(file_path, 'w') as file:
            json.dump(results, file)
            logger.info(f"New results save as '{file_path}' ")

        return results
    else:
        raise Exception('Error when evaluating the trained models')

def save_trained_models(trained_classifiers):

    for clf in trained_classifiers.keys():
        #Save trained model
        #DONE: add relative path
        print(f"BASE_DIR.parent.parent -> {globals.BASE_DIR.parent.parent}")
        print(f"BASE_DIR .parent.parent type -> {type(globals.BASE_DIR.parent.parent)}")

        #project_root = Path(__file__).resolve(strict=True).parent.parent.parent
        #print(f"project_root -> {project_root}")
        #data_path = os.path.join(project_root, "/data/iris-data.csv")
        models_dir = globals.BASE_DIR.parent.parent / "models/prod/"
        print(f"models_dir -> {models_dir}")
        print(f"models_dir type -> {type(models_dir)}")
    
        #models_dir = '../models/prod/'
        model_name = clf.replace(' ', '') + '.pkl'
        print(f"model_name -> {model_name}")
        print(f"model_name type -> {type(model_name)}")
    
        model_path = models_dir / model_name
        print(f"model_path -> {model_path}")
        print(f"model_path type -> {type(model_path)}")
    
        with open(model_path, 'wb') as model_file:
            pickle.dump(trained_classifiers[clf], model_file)
            logger.info(f"New Model {clf} saved!")
           

##################################
########### PREDICTION ###########
##################################

def predict(model_name:str, inputs:List[float]) -> List[int]:
    
    ### VALIDATE INPUT
    validate_input(inputs)
    ##data preprocessing

    preprocessed_input = input_preprocessing(inputs)
    print(f"preprocessed_input -> {preprocessed_input}")
    print(f"preprocessed_input type -> {type(preprocessed_input)}")
    print(f"preprocessed_input.shape -> {preprocessed_input.shape}")
    
    #validate and load model base on given name
    #base_dir = BASE_DIR
    #DONE: update to BASE_DIR
    print(f"BASE_DIR -> {globals.BASE_DIR.parent.parent}")
    print(f"BASE_DIR -> {type(globals.BASE_DIR.parent.parent)}")
    model_path = globals.BASE_DIR.parent.parent / f"models/prod/{model_name}.pkl"
    #model_path = Path(models_dir).joinpath(f"models/prod/{model_name}.pkl")
    print(f"model_path -> {model_path}")
    
    # base_dir = '/home/saito/Documents/picpay/iris-classifier-challenge/api/utils'
    # #print(f"base_dir-> {base_dir}")
    # base_dir = base_dir.replace("api/utils", "")
    # #print(f"base_dir-> {base_dir}")
    # model_path = Path(base_dir).joinpath(f"models/prod/{model_name}.pkl")
    # #print(f"model_path -> {model_path}")

    if not model_path.exists():
        raise Exception(f"Could not find model at {model_path}")
    
    with open(model_path, 'rb') as model_file:
        try:
            loaded_model = pickle.load(model_file)
            #print(f"loaded_model -> {loaded_model}")
        except:
            raise Exception('Error during the loading of the model')
        #Try to predict
        try:
            output = loaded_model.predict(preprocessed_input)
            #print(f"{model_name} ------->> prediction = {output}")
            #print(f"output len -> {len(output)}")
        except:
            raise Exception('Error during the prediction of the model')
        
        return output
    
    return []
   

def validate_input(inputs:List[float]):
    #DONE: validate inputs
    #check input type
    #float inputs lenght
    if not len(inputs) == 4:
        raise Exception("Inputs are expected to be list of 4 attributes (float)")
    #raise expection otherwise
    

def input_preprocessing(inputs:List[float]) -> pd.DataFrame:#np.array:
    #convert list to numpy array with shape 1,4
    #print(f" np.array(inputs) -> { np.array(inputs)}")
    column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    print(f"column_names -> {column_names}")
    #print(pd.DataFrame(inputs, columns=column_names).T)
    df = pd.DataFrame(inputs).T
    print(f"df -> {df}")
    print(f"df.columns  -> {df.columns }")
    df.columns = column_names
    #print(df.shape)
    print(f"df -> {df}")
    print(f"df -> {df.columns}")
    return df
    #return pd.DataFrame(inputs, columns=column_names).T
    #return np.array(inputs).reshape(1, -1)

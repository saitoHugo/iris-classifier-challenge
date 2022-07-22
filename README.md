# iris-classifier-challenge
Repository to train a Machine Learning models on Iris Specieis Public Dataset and deploy an FastAPI with basic endpoints using Docker at Heroku Cloud.

# Overview

## Dataset

![dataset](./assets/images/dataset-overview.png)

""
- The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
- Predicted attribute: class of iris plant
- Number of Instances: 150 (50 in each of three classes)
- Number of features (attributes): 4 numeric, predictive attributes and the class
- Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
       - Iris Setosa
       - Iris Versicolour
       - Iris Virginica
- Missing Attribute Values: None
- Class Distribution: 33.3% for each of 3 classes.

"" 

Fisher, R.A.. (1988). Iris. UCI Machine Learning Repository.



## Solution Approach

### 1) Problem and Data Exploration

- Problem Overview and Benchmarks
- Data Exploration
- Data Vizualization 

### 2) Model Tranining

- Data Preprocessing
- Models Experimentation
- Models Evaluation
- Twest Models Prediction 

### 3) API Development

- Routes Definition and Planning
- Framework Seletion (FastAPI)
- Endpoints Implementation
- API Local Test

### 4) Docker Container

- Dockerfile Config
- Build Docker Image
- Run Image Locally

### 5) Deploy Heroku

- Set up Heroku (login, new app, Procfile, yml)
- Registry Docker Image
- Perform First Release


## Repository Structure

    ```
    ├── .env                        # Local only files of enviroment variables
    ├── env-sample                  # Public files to list all necessary enviroment variables
    ├── .venv                       # Local only files of virtualenv 
    ├── .gitignore                  # File that tells Git which files or folders to ignore in a project
    ├── assets                      # Local storage for images and other static resources
    │   └── images                  # Folder to save images used in project 
    │
    ├── data                        # Local storage of the dataset
    │
    ├── iris-api                    # Main folder of FastAPI 
    │   ├── main                    # Main file fo API
    │   ├── utils                   # Tools and utilities (aux functions)
    │   └── tests                   # Automated tests 
    │   
    ├── models                      # Local storage of models trained 
    │   ├── dev                     # Local storage of DEV models  
    │   └── prod                    # Local storage of DEV models  
    │
    ├── notebooks                   # Folder to storage all jupyter notebooks
    │   ├── data_exploration.ipynb  #Data Exploration Proccess
    │   ├── training.ipynb  #Data Preparation and Training Proccess
    │ 
    │
    ├── Dockerfile                  # Dockerfile configuration
    │
    ├── Makefile                  # Make rules configuration
    │
    │
    └── README.md
    ```

# How to reproduce?

## Prerequisits

- Python - Version: 3.9.13
- Docker Engine Community - Version: 20.10.17
- Pip - Version:22.1.2
- Ubuntu OS 20.04


## Prepara the enviroment

- Create a new virtual env as follow:
    
    `python3 -m venv .venv`

- Activate enviroment
    
    `source .venv/bin/activate`

- Install requiments
    
    `python3 -m pip install --upgrade pip`

    `pip install -r requirements.txt`


## Run Local API

- After installing requirements, move to iris-api folder

    `cd iris-api/`
    

- Then, run the local server with uvicorn

    `uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8080`

- Or only run:
    `make dev`


### API Docs

- To access the api documentation access the link generated and you will redicrected to `/doc`:
    `http://0.0.0.0:8080`

**Endponits Disponíveis**

`/train` : used to execute a new training with all models

`/predict` :   used to execute a new prediction based on 4 features inputs


## Docker Build and Local Test

It's a prerequisite that you have docker installed.
- Go to the project directory (in where the Dockerfile and build your FastAPI image:
- Build the docker image running:
    `docker build --tag iris-api .`

- Or only run:
    `make docker-build`

- Run the docker image using:
    `docker run -i -d -p 8080:8080 iris-api`
    `docker run --publish 8080:8080 --name iris-api`
    `docker run -p 8080:8080 --name iris-api-container iris-api`

- Or only run:
    `make docker-run`


## Deploy

- heroku login

- heroku create iris-api-container

- heroku container:login

- Build the Docker image and tag it with the Heroku format:
    `make heroku-docker-build`

- Registry image in Heroku docker Registry:
    `make heroku-docker-registry`

- Perform a release:
    `make heroku-release`

- Now a new deploy is executed in Heroku, acess public link:
    https://iris-classifier-challenge.herokuapp.com/

# Next Step and Improvements


## Major Updates

- Add Initial Unit Test

- Start Use TDD

- Split Training phase into more microsservices
    - train()

        - /get_data
        - /data pipeline
        - /train_model(model_name, test_size, cross_validation, grid_search)
        - /evaluate-model(model_name, train_score, acc_score)
        - /validate_model(model_name)
        - /update-new-model()

- Training Updates

    - Cross Validation: applied to deal variance problem based on small toy dataset. It also would be useful to define the best model.

    - Grid Search: applied optimize the hyper-parameters I could have used GridSearch for find the best models configurations for the toy dataset. It also would be useful to define the best model.


- Define one best model
    - load model on api startup
    - update loaded model after new trainining

- Add MongoDB Database
    - add static raw dataset to a bucket 
    - update data transformation and save data transformed
    - update training to get data from db

## Further Improvements

- Use MlFlow or Kubeflow to manage ML Model lifecycle
- Definir pipeline para ingestão e processamento de novos dados para o banco (feat Store) de forma isolada
- Versionmaneto data, training, model and perfmomance
    - artifact
- Monitoramento do modelo
    - alarms, planos de incidencia
    - health system
    - validation of new data
    - prediction probs
        - change based on previous predictions
        - change of feat distribution
        - change on prediction classes
        - group by specieis prediction
  
- Define a pipepline for re-treino and when to execute

- Melhorias de Infra
    - kubernets (Kubeflow Pipelines)
- treinamento em cluster separados
- Ambiente de homologação e experimentação

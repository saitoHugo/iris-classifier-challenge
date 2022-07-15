# iris-classifier-challenge
Repository to train a Machine Learning model on Iris Public Dataset and deploy an FastAPI with basic endpoints using Docker at AWS Cloud.

# TODO

- add iris challenge overview
- add solution approach and steps
- add refs used



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

# Initial Setup

## Prerequisits

- Python3 3.9.13
- Pip 22.1.2
- Ubuntu OS


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


### API Docs

- To access the api documentation access the link generated and you will redicrected to `/doc`:
    `http://0.0.0.0:8080`

**Endponits Disponíveis**

`/train` : used to execute a new training with all models

`/predict` :   used to execute a new prediction based on 4 features inputs


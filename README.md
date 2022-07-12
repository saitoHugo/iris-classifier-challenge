# iris-classifier-challenge
Repository to train a Machine Learning model on Iris Public Dataset and deploy an FastAPI with basic endpoints using Docker at AWS Cloud.

## Repository Structure

Section to exaplain the repository structure.


#Initial Setup

## Prerequisits

- Python 3.9.13

## Prepara the enviroment

- Create a new virtual env as follow:
    `python3 -m venv .venv`
- Activate enviroment
    `source .venv/bin/activate`
- Install requiments
    `python3 -m pip install --upgrade pip`
    `pip install -r requirements.txt`

## Run Local API

- After installing requirements, run the server with uvicorn
    `uvicorn main:app --reload`
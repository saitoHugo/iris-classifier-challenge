from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pathlib import Path

app = FastAPI()


"""
TODO: 
    - load all models gloabally
    - create a train route
    - create a predict route based on model selection
    - extra route get_all_trained_models
    - download local data or download from bucket
"""

BASE_DIR = Path(__file__).resolve(strict=True).parent


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')


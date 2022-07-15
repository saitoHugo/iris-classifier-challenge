from pathlib import Path
"""
File to save all global variables declared in api
"""
#define root path
global BASE_DIR
BASE_DIR = Path(__file__).resolve(strict=True).parent


#load dataset globally

#load all model glabally if exists
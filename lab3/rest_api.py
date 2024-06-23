from fastapi import FastAPI

from lab3.data import download_and_save_data
from lab3.entity import IrisRequest
from lab3.model import processing, check_result
from lab3.test_model import get_model, get_predict


app = FastAPI()
download_and_save_data()
processing()
check_result()
model = get_model()


@app.post("/predict")
def get_history(iris_request: IrisRequest):
    """Запрос истории ответов текущей сессии."""
    result = get_predict(model, iris_request)
    return result

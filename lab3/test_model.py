import pandas as pd
import pickle

from lab3.entity import IrisRequest, IrisPredictEntity, IrisPredictResponse

model_path = './model.pkl'


def get_model():
    return pickle.load(open(model_path, 'rb'))


def get_predict(model, data: IrisRequest):
    response_list = list()
    for el in data.iris_entities:
        df = pd.DataFrame([[
            el['sepal_len'],
            el['sepal_wid'],
            el['petal_len'],
            el['petal_wid']]])
        prediction = model.predict(df)

        response_row = IrisPredictEntity(
            el['sepal_len'],
            el['sepal_wid'],
            el['petal_len'],
            el['petal_wid'],
            prediction[0]
        )
        response_list.append(response_row)
    return IrisPredictResponse(response_list)

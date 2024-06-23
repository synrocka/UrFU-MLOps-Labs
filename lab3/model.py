import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pickle
import os


model_path = "./model.pkl"
data_path = "./iris.csv"


def processing():
    # DATA PROCESSING
    iris = pd.read_csv(data_path)
    X = iris.drop("Species", axis=1)
    y = iris["Species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # SCALING
    scaler = StandardScaler()
    # scaled_X_train = scaler.fit_transform(X_train)
    # scaled_X_test = scaler.transform(X_test)

    # MODEL
    model = LogisticRegression()
    # model.fit(scaled_X_train, y_train)

    # PIPELINE
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", model)])
    pipe.fit(X_train, y_train)

    # METRICS
    y_predict = pipe.predict(X_test)
    print(f"Metrics:\n {classification_report(y_test, y_predict)}")

    # SAVE MODEL
    pickle.dump(model, open(model_path, 'wb'))


def check_result():
    # CHECK RESULT
    if os.path.exists(model_path):
        print("Successful model saving")
    else:
        raise FileNotFoundError("Something wrong with model saving")

    # DELETE DATA FILE
    os.remove(data_path)

    # CHECK RESULT OF DELETING
    if not os.path.exists(data_path):
        print("Successful data removing")
    else:
        raise FileNotFoundError("Something wrong with data removing")


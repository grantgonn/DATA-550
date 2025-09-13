import pandas as pd
import numpy as np
import mlflow
import json

from sklearn.ensemble import RandomForestRegressor
from evidently.dashboard import Dashboard
from evidently.tabs import RegressionPerformanceTab

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

np.object = object

# Define features
reference = train.drop(columns=['Age'], axis=1)
y = train['Age']
production = test.drop(columns=['Age'], axis=1)
y_test = test['Age']

# loading RF model
with open("RF_params.json", "r") as f:
    RF_params = json.load(f)
    
RF_model = RandomForestRegressor(**RF_params)

Experiment_name = 'HW7'
mlflow.set_experiment(Experiment_name)

with mlflow.start_run():
    # fit model
    RF_model.fit(reference, y)
    
    # adding dataset column to distinguish between reference and production data
    reference['dataset'] = 'reference'
    production['dataset'] = 'production'
    
    # compute predictions
    reference['target'] = y
    reference['prediction'] = RF_model.predict(reference.drop(columns=['dataset', 'target'], axis=1))
    
    production['target'] = y_test
    production['prediction'] = RF_model.predict(production.drop(columns=['dataset', 'target'], axis=1))
    
    # define column mapping for evidently
    column_mapping = {'target': 'target', 'prediction': 'prediction', 'numerical_features': list(reference.select_dtypes(include=['number']).columns), 'categorical_features': list(reference.select_dtypes(include=['object']).columns)}

    # create dashboard
    dashboard = Dashboard(tabs=[RegressionPerformanceTab])
    
    # calculate drift
    dashboard.calculate(reference, production, column_mapping=column_mapping)
    
    # save report
    dashboard.save('HW7_model_drift.html')
    
    # log report to mlflow
    mlflow.log_artifact('HW7_model_drift.html')


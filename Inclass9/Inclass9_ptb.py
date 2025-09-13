import pandas as pd
import numpy as np
import mlflow

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

# Load the data
ref_df = pd.read_csv('train.csv')
latest_df = pd.read_csv('test.csv')

# define the exeriement name
Experiment_name = 'Inclass9'
mlflow.set_experiment(Experiment_name)

np.object = object

with mlflow.start_run():
    # create the data drift dashboard
    dashboard = Dashboard(tabs=[DataDriftTab])
    
    # calculate the data drift
    dashboard.calculate(ref_df, latest_df)
    
    # save the dashboard
    dashboard.save('Inclass9_data_drift_dashboard.html')
    
    # log the dashboard
    mlflow.log_artifact('Inclass9_data_drift_dashboard.html')
import pandas as pd
import mlflow
import numpy as np

from sklearn.metrics import root_mean_squared_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('charges', axis=1)
y = df['charges']

# Load the rf model
logged_model = 'runs:/8c6d26e0c9384a8c86fc6eea7e493774/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/bf130c6a4a894efd8c7ee69f89f28b9b/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/7ba0fc81574642558c8cb7b794c5f5e0/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/dedbcbffc7674f338117add73ed166e1/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

#load the vote model
logged_model = 'runs:/fb159776c59a41e6aa6b2847dd0f7bc6/model'
vote_model = mlflow.pyfunc.load_model(logged_model)

#load the stack model
logged_model = 'runs:/33e8fe7f1ac24048b1c31974a82f0353/model'
stack_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)
vote_pred = vote_model.predict(X)
stack_pred = stack_model.predict(X)

# Calculate the mae
rf_mae = root_mean_squared_error(y, rf_pred)
print(f'Random Forest mae: {rf_mae}')
et_mae = root_mean_squared_error(y, et_pred)
print(f'Extra Trees mae: {et_mae}')
gb_mae = root_mean_squared_error(y, gb_pred)
print(f'Gradient Boosting mae: {gb_mae}')
xgb_mae = root_mean_squared_error(y, xgb_pred)
print(f'XGBoost mae: {xgb_mae}')
vote_mae = root_mean_squared_error(y, vote_pred)
print(f'Vote mae: {vote_mae}')
stack_mae = root_mean_squared_error(y, stack_pred)
print(f'Stack mae: {stack_mae}')
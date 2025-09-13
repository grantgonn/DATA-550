import pandas as pd
import mlflow
import numpy as np

from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('Rented Bike Count', axis=1)
y = df['Rented Bike Count']

# Load the rf model
logged_model = 'runs:/5b1857b5b9cf4239b19785119fd8ac95/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/b7558ccaa46c448bbc5deb22c85a3351/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/481c03c2198b4023bc9473556ced9a8f/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/94bc61570da149638f6f71a53f8cdf2a/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

#load the lgb model
logged_model = 'runs:/e7b5cde6b6c840df868d465759c0f576/model'
lgb_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)
lgb_pred = lgb_model.predict(X)

# Calculate the rmse
rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))
print(f'Random Forest rmse: {rf_rmse}')
et_rmse = np.sqrt(mean_squared_error(y, et_pred))
print(f'Extra Trees rmse: {et_rmse}')
gb_rmse = np.sqrt(mean_squared_error(y, gb_pred))
print(f'Gradient Boosting rmse: {gb_rmse}')
xgb_rmse = np.sqrt(mean_squared_error(y, xgb_pred))
print(f'XGBoost rmse: {xgb_rmse}')
lgb_rmse = np.sqrt(mean_squared_error(y, lgb_pred))
print(f'LightGBM rmse: {lgb_rmse}')

avg_pred = (rf_pred + et_pred + gb_pred + xgb_pred + lgb_pred) / 5
avg_rmse = np.sqrt(mean_squared_error(y, avg_pred))
print(f'Average rmse: {avg_rmse}')
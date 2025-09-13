import pandas as pd
import mlflow

from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('charges', axis=1)
y = df['charges']

# Load the rf model
logged_model = 'runs:/d10d3614231c4933a31104a617c1b204/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/6303c038c09142c39823e5c8a19a804c/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/dc9d688d60b14f92bc45d002490646eb/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/4cfb28e4d8b24c5b8c275f8db1e873ed/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)

# Calculate the rmse
rf_rmse = mean_squared_error(y, rf_pred, squared=False)
print(f'Random Forest RMSE: {rf_rmse}')
et_rmse = mean_squared_error(y, et_pred, squared=False)
print(f'Extra Trees RMSE: {et_rmse}')
gb_rmse = mean_squared_error(y, gb_pred, squared=False)
print(f'Gradient Boosting RMSE: {gb_rmse}')
xgb_rmse = mean_squared_error(y, xgb_pred, squared=False)
print(f'XGBoost RMSE: {xgb_rmse}')

avg_pred = (rf_pred + et_pred + gb_pred + xgb_pred) / 4
avg_rmse = mean_squared_error(y, avg_pred, squared=False)
print(f'Average RMSE: {avg_rmse}')
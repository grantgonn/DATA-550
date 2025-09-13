import pandas as pd
import mlflow

from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('charges', axis=1)
y = df['charges']

# Load the rf model
logged_model = 'runs:/7c11439fe9794e3498fcdb1ed09fea1b/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/c51fa104f02f4301ad7096d4ec93d074/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/460a5b52711f45a1a1a815c84eacdad1/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/fc78d75621404eae996e7e7dc15cb477/model'
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
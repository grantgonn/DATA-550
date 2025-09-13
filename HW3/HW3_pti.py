import pandas as pd
import mlflow

from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('Age', axis=1)
y = df['Age']

# Load the rf model
logged_model = 'runs:/db480050a83542819919085e9bf3b13a/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/5cdb32dfa6a343e5bb2de3a696e3f858/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/6641241f293b4510ae21dcbf436e33d6/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/d8f0ea84ae9b4bcfa91b7c74c9542676/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

#load the lgb model
logged_model = 'runs:/7a28f502eb3e4480b299ca48560a0172/model'
lgb_model = mlflow.pyfunc.load_model(logged_model)

#load the cat model
logged_model = 'runs:/cb24c6271dc54ae38fa573c9be0b88de/model'
cat_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)
lgb_pred = lgb_model.predict(X)
cat_pred = cat_model.predict(X)

# Calculate the mae
rf_mae = mean_absolute_error(y, rf_pred)
print(f'Random Forest mae: {rf_mae}')
et_mae = mean_absolute_error(y, et_pred)
print(f'Extra Trees mae: {et_mae}')
gb_mae = mean_absolute_error(y, gb_pred)
print(f'Gradient Boosting mae: {gb_mae}')
xgb_mae = mean_absolute_error(y, xgb_pred)
print(f'XGBoost mae: {xgb_mae}')
lgb_mae = mean_absolute_error(y, lgb_pred)
print(f'LightGBM mae: {lgb_mae}')
cat_mae = mean_absolute_error(y, cat_pred)
print(f'CatBoost mae: {cat_mae}')

avg_pred = (rf_pred + et_pred + gb_pred + xgb_pred + lgb_pred + cat_pred) / 6
avg_mae = mean_absolute_error(y, avg_pred)
print(f'Average mae: {avg_mae}')
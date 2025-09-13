import pandas as pd
import mlflow
import numpy as np

from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('Age', axis=1)
y = df['Age']

# Load the rf model
logged_model = 'runs:/967876d189e648e38735480377a35c12/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/9948fe7e56b143c7b30e98e697cb689e/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/b456f1442f0c47b7bd0f0060fcf2e838/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/efa1aaf035a648489de96148f502f489/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

#load the lgb model
logged_model = 'runs:/ef1c84db515a41348b9be4d6ac9daf77/model'
lgb_model = mlflow.pyfunc.load_model(logged_model)

#load the lgb model
logged_model = 'runs:/5b0461d5a18e4a6f8e91e4a9eaea5e0a/model'
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

# Calculate ensemble mae
avg_pred = (rf_pred + et_pred + gb_pred + xgb_pred + lgb_pred + cat_pred) / 6
avg_mae = mean_absolute_error(y, avg_pred)
print(f'Average mae: {avg_mae}')

# Calculate weighted ensemble mae
wtot = rf_mae + et_mae + gb_mae + xgb_mae + lgb_mae + cat_mae
rf_wt = rf_mae / wtot
et_wt = et_mae / wtot
gb_wt = gb_mae / wtot
xgb_wt = xgb_mae / wtot
lgb_wt = lgb_mae / wtot
cat_wt = cat_mae / wtot

wt_avg_pred = (rf_pred * rf_wt + et_pred * et_wt + gb_pred * gb_wt + xgb_pred * xgb_wt + lgb_pred * lgb_wt + cat_pred * cat_wt)
wt_avg_mae = mean_absolute_error(y, wt_avg_pred)
print(f'Weighted Average mae: {wt_avg_mae}')
import pandas as pd
import mlflow
import numpy as np

from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('Age', axis=1)
y = df['Age']

# Load the rf model
logged_model = 'runs:/33273386296146fdad365b805e3e068e/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/9c409cc93d574a4db6fc44f101317630/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/896f85881f49432b9b0eab560b3fc565/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/5f1d904a7b5a4891bf9930b843023652/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

# load the lgb model
logged_model = 'runs:/df0efa091f9c4f459d5b1e3850dbc41f/model'
lgb_model = mlflow.pyfunc.load_model(logged_model)

# load the cat model
logged_model = 'runs:/e935d29b355c4644a6a90943ab96f534/model'
cat_model = mlflow.pyfunc.load_model(logged_model)

#load the vote model optuna
logged_model = 'runs:/98b6f2c43ecd4d879160f04803796bd8/model'
vote_model_optuna = mlflow.pyfunc.load_model(logged_model)

#load the vote model
logged_model = 'runs:/012ae74aad18463285fac15d4c2f36f3/model'
vote_model = mlflow.pyfunc.load_model(logged_model)

#load the stack model mlp
logged_model = 'runs:/cad5020fea3b418d883f814e02dc7b43/model'
stack_model_mlp = mlflow.pyfunc.load_model(logged_model)

#load the stack model
logged_model = 'runs:/cd31a797f2884f6cb4cb080e406c3414/model'
stack_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)
lgb_pred = lgb_model.predict(X)
cat_pred = cat_model.predict(X)
vote_pred_optuna = vote_model_optuna.predict(X)
vote_pred = vote_model.predict(X)
stack_pred_mlp = stack_model_mlp.predict(X)
stack_pred = stack_model.predict(X)

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
vote_optuna_mae = mean_absolute_error(y, vote_pred_optuna)
print(f'Vote optuna mae: {vote_optuna_mae}')
vote_mae = mean_absolute_error(y, vote_pred)
print(f'Vote mae: {vote_mae}')
stack_mlp_mae = mean_absolute_error(y, stack_pred_mlp)
print(f'Stack MLP mae: {stack_mlp_mae}')
stack_mae = mean_absolute_error(y, stack_pred)
print(f'Stack mae: {stack_mae}')

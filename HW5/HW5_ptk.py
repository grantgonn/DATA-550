import pandas as pd
import mlflow
import numpy as np

from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('Age', axis=1)
y = df['Age']

# Load the rf model
logged_model = 'runs:/e8b803ac2dd440d2a34459c4db4e0a71/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/3d92d77352294ce19adee20743cb7d95/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/c766a28032604c21af8cb2d743351bd2/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/4d554449de9b4fe0971132b72fb30988/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

# load the lgb model
logged_model = 'runs:/c695691224704a278c58885648bd0b67/model'
lgb_model = mlflow.pyfunc.load_model(logged_model)

# load the cat model
logged_model = 'runs:/643ecdbe85cf49bdbedfbb488c251ce5/model'
cat_model = mlflow.pyfunc.load_model(logged_model)

#load the vote model
logged_model = 'runs:/c8f6cd8cd7b642b99e399753a755050b/model'
vote_model = mlflow.pyfunc.load_model(logged_model)

#load the stack model
logged_model = 'runs:/76f86f6370fe4028bc1ff235a8d8e4d0/model'
stack_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)
lgb_pred = lgb_model.predict(X)
cat_pred = cat_model.predict(X)
vote_pred = vote_model.predict(X)
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
vote_mae = mean_absolute_error(y, vote_pred)
print(f'Vote mae: {vote_mae}')
stack_mae = mean_absolute_error(y, stack_pred)
print(f'Stack mae: {stack_mae}')
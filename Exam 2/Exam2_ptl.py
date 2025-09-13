import pandas as pd
import mlflow
import numpy as np

from sklearn.metrics import root_mean_squared_error

# Load the data
df = pd.read_csv('test.csv')
X = df.drop('Rented Bike Count', axis=1)
y = df['Rented Bike Count']

# Load the rf model
logged_model = 'runs:/71a66c7d284144608e21bfc825b2f2bc/model'
rf_model = mlflow.pyfunc.load_model(logged_model)

# load the et model
logged_model = 'runs:/657d70e3086c490bab56f5f4409061a9/model'
et_model = mlflow.pyfunc.load_model(logged_model)

# load the gb model
logged_model = 'runs:/35bd93e50b994083af84586b6783f0cf/model'
gb_model = mlflow.pyfunc.load_model(logged_model)

# load the xgb model
logged_model = 'runs:/df27509cef61470e83b58a3ef09ed8ba/model'
xgb_model = mlflow.pyfunc.load_model(logged_model)

#load the lgb model
logged_model = 'runs:/aa66b5a190764a60b88ac0e14c388e40/model'
lgb_model = mlflow.pyfunc.load_model(logged_model)

#load the cat model
logged_model = 'runs:/e1ab57483c0142dbbba4b8c551516f54/model'
cat_model = mlflow.pyfunc.load_model(logged_model)

#load the voting model
logged_model = 'runs:/1a3a54deeca0434a916121f1869ebba9/model'
voting_model = mlflow.pyfunc.load_model(logged_model)

#load the stacking model
logged_model = 'runs:/ebce032b132d47b6a5f5327a710c3488/model'
stacking_model = mlflow.pyfunc.load_model(logged_model)

# Predict the test data
rf_pred = rf_model.predict(X)
et_pred = et_model.predict(X)
gb_pred = gb_model.predict(X)
xgb_pred = xgb_model.predict(X)
lgb_pred = lgb_model.predict(X)
cat_pred = cat_model.predict(X)
voting_pred = voting_model.predict(X)
stacking_pred = stacking_model.predict(X)

# Calculate the rmse
rf_rmse = root_mean_squared_error(y, rf_pred)
print(f'Random Forest rmse: {rf_rmse}')
et_rmse = root_mean_squared_error(y, et_pred)
print(f'Extra Trees rmse: {et_rmse}')
gb_rmse = root_mean_squared_error(y, gb_pred)
print(f'Gradient Boosting rmse: {gb_rmse}')
xgb_rmse = root_mean_squared_error(y, xgb_pred)
print(f'XGBoost rmse: {xgb_rmse}')
lgb_rmse = root_mean_squared_error(y, lgb_pred)
print(f'LightGBM rmse: {lgb_rmse}')
cat_rmse = root_mean_squared_error(y, cat_pred)
print(f'CatBoost rmse: {cat_rmse}')
voting_rmse = root_mean_squared_error(y, voting_pred)
print(f'Voting rmse: {voting_rmse}')
stacking_rmse = root_mean_squared_error(y, stacking_pred)
print(f'Stacking rmse: {stacking_rmse}')

avg_pred = (rf_pred + et_pred + gb_pred + xgb_pred + lgb_pred + cat_pred + voting_pred + stacking_pred) / 8
avg_rmse = root_mean_squared_error(y, avg_pred)
print(f'Average rmse: {avg_rmse}')
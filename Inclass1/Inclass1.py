import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# Load data
df = pd.read_csv('insurance.csv')

# define x and y
x = df.drop('charges', axis=1)
x = pd.get_dummies(columns=['sex', 'smoker', 'region'], data=x, drop_first=True, dtype=int)

y = df['charges']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# setting the experiment
mlflow.set_experiment('Grantg-Insurance-Regression')

#Start run
with mlflow.start_run():
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_cv = cross_val_score(rf, x, y, cv=kf, scoring='neg_mean_squared_error')
    rf_mse = -1.0*rf_cv.mean()
    print(f'Random Forest: {rf_mse}')
    
    #logging the rf results
    mlflow.log_param('rf_model', 'Random Forest')
    mlflow.log_metric('rf_mse', rf_mse)
    mlflow.sklearn.log_model(rf, 'rf_model')
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_cv = cross_val_score(gb, x, y, cv=kf, scoring='neg_mean_squared_error')
    gb_mse = -1.0*gb_cv.mean()
    print(f'Gradient Boosting: {gb_mse}')
    
    #logging the gb results
    mlflow.log_param('gb_model', 'Gradient Boosting')
    mlflow.log_metric('gb_mse', gb_mse)
    mlflow.sklearn.log_model(gb, 'gb_model')
    
    # Extra Trees
    et = ExtraTreesRegressor()
    et_cv = cross_val_score(et, x, y, cv=kf, scoring='neg_mean_squared_error')
    et_mse = -1.0*et_cv.mean()
    print(f'Extra Trees: {et_mse}')
    
    #logging the et results
    mlflow.log_param('et_model', 'Extra Trees')
    mlflow.log_metric('et_mse', et_mse)
    mlflow.sklearn.log_model(et, 'et_model')
    




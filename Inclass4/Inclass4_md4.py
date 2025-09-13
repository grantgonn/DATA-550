import pandas as pd 
import mlflow

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

import optuna

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('charges', axis=1)
y = df['charges']

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    # Define the hyperparameters
    params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
                    random_state=trial.suggest_categorical('random_state', [42]))

    # Create a GrandientBoostingRegressor
    md = XGBRegressor(**params)

    # Train the model using KFold
    scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, n_jobs=-1)
XGB_params = study.best_params
XGB_rmse = study.best_value

mlflow.set_experiment('Inclass4')
mlflow.sklearn.autolog()

with mlflow.start_run(run_name='XGB Optuna') as run:
    #log best params
    mlflow.log_params(XGB_params)
    
    #define the model
    model = XGBRegressor(**XGB_params)
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('rmse', XGB_rmse)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'In class 4 XGB Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'Gradient Boosting',
                         'feature_version': 1})
    mlflow.end_run()
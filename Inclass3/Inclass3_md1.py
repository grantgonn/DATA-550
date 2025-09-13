import pandas as pd 
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

import optuna

# Load the data
df = pd.read_csv('insurance.csv')

X = df.drop('charges', axis=1)
y = df['charges']

X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True, dtype='int')

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

# setting mlflow
mlflow.set_experiment('Inclass3')

def objective(trial):
    with mlflow.start_run(nested=True):
        # Define the hyperparameters
        params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 300),
                        max_depth=trial.suggest_int('max_depth', 5, 10),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                        random_state=trial.suggest_categorical('random_state', [42]),
                        n_jobs=trial.suggest_categorical('n_jobs', [-1]))

        # Create a RandomForestClassifier
        md = RandomForestRegressor(**params)

        # Train the model using KFold
        scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_root_mean_squared_error')

        mlflow.log_params(params)
        mlflow.log_metric('rmse', scores)
    return scores


with mlflow.start_run(run_name='RF Optuna') as run:
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    #log best params
    best_params = study.best_params
    mlflow.log_params(best_params)
    
    # log score
    mlflow.log_metric('rmse', study.best_value)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'In class 3 RF Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'Random forest',
                         'feature_version': 1})
    mlflow.end_run()

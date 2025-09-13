import pandas as pd 
import mlflow

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score

import optuna

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Age', axis=1)
y = df['Age']

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 10),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=trial.suggest_categorical('random_state', [42]),
                    n_jobs=trial.suggest_categorical('n_jobs', [-1]))

    # Create a ExtraTreesRegressor
    md = ExtraTreesRegressor(**params)

    # Train the model using KFold
    scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, n_jobs=-1)
ET_params = study.best_params
ET_mae = study.best_value

mlflow.set_experiment('HW3')
mlflow.sklearn.autolog()

with mlflow.start_run(run_name='ET Optuna') as run:
    #log best params
    mlflow.log_params(ET_params)
    
    #define the model
    model = ExtraTreesRegressor(**ET_params)
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('mae', ET_mae)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'HW3 ET Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'Extra Trees',
                         'feature_version': 1})
    mlflow.end_run()
import pandas as pd 
import mlflow

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score

import optuna

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Age', axis=1)
y = df['Age']

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    # Define the hyperparameters
    params = dict(iterations=trial.suggest_int('iterations', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 10),
                    verbose=False)

    # Create a CatBoostRegressor
    md = CatBoostRegressor(**params)

    # Train the model using KFold
    scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, n_jobs=-1)
CAT_params = study.best_params
CAT_mae = study.best_value

mlflow.set_experiment('HW3')
mlflow.sklearn.autolog()

with mlflow.start_run(run_name='CAT Optuna') as run:
    #log best params
    mlflow.log_params(CAT_params)
    
    #define the model
    model = CatBoostRegressor(**CAT_params)
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('mae', CAT_mae)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'HW3 CAT Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'CAT Boost',
                         'feature_version': 1})
    mlflow.end_run()
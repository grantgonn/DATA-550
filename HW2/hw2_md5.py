import pandas as pd 
import mlflow

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score

import optuna

# Load the data
df = pd.read_csv('CrabAgePrediction.csv')

X = df.drop('Age', axis=1)
y = df['Age']

X = pd.get_dummies(X, columns=['Sex'], drop_first=True, dtype='int')

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

# setting mlflow
mlflow.set_experiment('HW2')

# Define the objective function
def objective(trial):

    with mlflow.start_run(nested=True):

        # Define the hyperparameters to optimize
        params = dict(iterations=trial.suggest_int("iterations", 100, 300),
                    max_depth=trial.suggest_int("max_depth", 5, 10),
                    verbose=False)

        # Define the model
        model = CatBoostRegressor(**params)

        # Compute the cross-validation score
        score = -1*cross_val_score(model, X, y, cv=skf, scoring="neg_root_mean_squared_error").mean()

        # Log the hyperparameters
        mlflow.log_params(params)
        mlflow.log_metric("rmse", score)

    return score

# Define the study
with mlflow.start_run(run_name="Cat Optuna") as run:

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Log the best hyperparameters
    best_params = study.best_params
    mlflow.log_params(best_params)

    # Log the best score
    mlflow.log_metric("rmse", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "HW2 Cat Optuna",
            "optimizer": "Optuna",
            "model_family": "Cat Boosting",
            "feature_version": 1,
        }
    )
    mlflow.end_run()

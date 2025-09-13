import pandas as pd 
import mlflow

from sklearn.ensemble import RandomForestRegressor
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
        params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    max_depth=trial.suggest_int("max_depth", 5, 10),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=trial.suggest_categorical("random_state", [42]),
                    n_jobs=trial.suggest_categorical("n_jobs", [-1]))

        # Define the model
        model = RandomForestRegressor(**params)

        # Compute the cross-validation score
        score = -1*cross_val_score(model, X, y, cv=skf, scoring="neg_root_mean_squared_error").mean()

        # Log the hyperparameters
        mlflow.log_params(params)
        mlflow.log_metric("rmse", score)

    return score

# Define the study
with mlflow.start_run(run_name="RF Optuna") as run:

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
            "project": "HW2 RF Optuna",
            "optimizer": "Optuna",
            "model_family": "Random Forest",
            "feature_version": 1,
        }
    )
    mlflow.end_run()

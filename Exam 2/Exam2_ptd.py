import pandas as pd 
import mlflow
import json

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance

import optuna

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Rented Bike Count', axis=1)
y = df['Rented Bike Count']

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial, X, y):
    params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
                    random_state=trial.suggest_categorical('random_state', [42]))

    # Create a GradientBoostingRegressor
    md = GradientBoostingRegressor(**params)

    # Train the model using KFold
    scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=30, n_jobs=-1)
GB_params = study.best_params
GB_rmse = study.best_value

# save the best params
with open('GB_params.json', 'w') as f:
    json.dump(GB_params, f)

mlflow.set_experiment('Exam2')
#mlflow.sklearn.autolog()

with mlflow.start_run(run_name='GB Optuna') as run:
    #log best params
    mlflow.log_params(GB_params)
    
    #define the model
    model = GradientBoostingRegressor(**GB_params)
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('rmse', GB_rmse)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'Exam 2 GB Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'Gradient Boosting',
                         'feature_version': 1})
    mlflow.end_run()
    
# compute permutation importance
model = GradientBoostingRegressor(**GB_params).fit(X, y)
perm_scores = permutation_importance(model, X, y, n_repeats=20, random_state=42, n_jobs=-1)

with mlflow.start_run(run_name='GB Optuna FS') as run:
    for i in range(5,17):
        # select top n features
        top_n_features = (X.columns[perm_scores.importances_mean.argsort()[::-1][:i]])

        # defining the model with top 5 features
        X_new = X[top_n_features]
        
        #define the model
        model_new = GradientBoostingRegressor(**GB_params)
        
        n_rmse = -1* cross_val_score(model_new, X_new, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        
        if n_rmse < GB_rmse:    
            #fit the model
            model_new.fit(X_new, y)
            mlflow.sklearn.log_model(model_new, 'model', input_example=X_new.head())
            mlflow.log_metric('rmse', n_rmse)
            mlflow.sklearn.log_model(model_new, artifact_path=f"model_top_{i}")

            # log tags
            mlflow.set_tags(tags={'Project': 'Exam 2 GB Optuna FS',
                                'Opimizer': 'Optuna',
                                'Model_family': 'Gradient Boosting',
                                'feature_version': 2})
            mlflow.end_run()
            print(f"Model with top {i} features outperformed original. RMSE: {n_rmse:.4f}")
            break
    else:
        print('No smaller rmse found with less features')
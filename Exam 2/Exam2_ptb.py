import pandas as pd 
import mlflow
import json

from sklearn.ensemble import RandomForestRegressor
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
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=trial.suggest_categorical('random_state', [42]),
                    n_jobs=trial.suggest_categorical('n_jobs', [-1]))

    # Create a RandomForestClassifier
    md = RandomForestRegressor(**params)

    # Train the model using KFold
    scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=30, n_jobs=-1)
RF_params = study.best_params
RF_rmse = study.best_value

# save the best params
with open('RF_params.json', 'w') as f:
    json.dump(RF_params, f)

mlflow.set_experiment('Exam2')
#mlflow.sklearn.autolog()

with mlflow.start_run(run_name='RF Optuna') as run:
    #log best params
    mlflow.log_params(RF_params)
    
    #define the model
    model = RandomForestRegressor(**RF_params)
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('rmse', RF_rmse)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'Exam 2 RF Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'Random forest',
                         'feature_version': 1})
    mlflow.end_run()
    
# compute permutation importance
model = RandomForestRegressor(**RF_params).fit(X, y)
perm_scores = permutation_importance(model, X, y, n_repeats=20, random_state=42, n_jobs=-1)

with mlflow.start_run(run_name='RF Optuna FS') as run:
    for i in range(5,17):
        # select top n features
        top_n_features = (X.columns[perm_scores.importances_mean.argsort()[::-1][:i]])

        # defining the model with top 5 features
        X_new = X[top_n_features]
        
        #define the model
        model_new = RandomForestRegressor(**RF_params)
        
        n_rmse = -1* cross_val_score(model_new, X_new, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        
        if n_rmse < RF_rmse:    
            #fit the model
            model_new.fit(X_new, y)
            mlflow.sklearn.log_model(model_new, 'model', input_example=X_new.head())
            mlflow.log_metric('rmse', n_rmse)
            mlflow.sklearn.log_model(model_new, artifact_path=f"model_top_{i}")

            # log tags
            mlflow.set_tags(tags={'Project': 'Exam 2 RF Optuna FS',
                                'Opimizer': 'Optuma',
                                'Model_family': 'Random forest',
                                'feature_version': 2})
            mlflow.end_run()
            print(f"Model with top {i} features outperformed original. RMSE: {n_rmse:.4f}")
            break
    else:
        print('No smaller rmse found with less features')
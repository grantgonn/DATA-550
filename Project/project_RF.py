import pandas as pd 
import mlflow
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

import optuna

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# map the target 2 to 1
y = y.map({0: 0, 1: 1, 2: 1})

# define kfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def find_best_threshold(model, X, y, average='macro'):
    y_proba = model.predict_proba(X)[:, 1]  # Probabilities for class 1
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y, y_pred, average=average)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1

def objective(trial, X, y):
    params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 10),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=trial.suggest_categorical('random_state', [42]),
                    n_jobs=trial.suggest_categorical('n_jobs', [-1]))

    # Create a RandomForestClassifier
    md = RandomForestClassifier(**params)

    # Train the model using KFold
    scores = cross_val_score(md, X, y, cv=skf, scoring='f1_macro', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=30, n_jobs=-1)
RF_params = study.best_params
RF_f1 = study.best_value

# save the best params
with open('RF_params.json', 'w') as f:
    json.dump(RF_params, f)

mlflow.set_experiment('project')
with mlflow.start_run(run_name='RF Optuna') as run:
    #log best params
    mlflow.log_params(RF_params)
    
    #define the model
    model = RandomForestClassifier(**RF_params)
    
    # Fit the model
    model.fit(X, y)

    # Find the best threshold
    best_threshold, threshold_f1 = find_best_threshold(model, X, y)

    # Log the model
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())

    # Log metrics
    mlflow.log_metric('optuna_cv_f1', RF_f1)
    mlflow.log_metric('best_threshold', best_threshold)
    mlflow.log_metric('threshold_f1', threshold_f1)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'Project RF Optuna',
                         'Opimizer': 'Optuna',
                         'Model_family': 'Random forest',
                         'feature_version': 1})
    mlflow.end_run()
    
# compute permutation importance
model = RandomForestClassifier(**RF_params).fit(X, y)
perm_scores = permutation_importance(model, X, y, n_repeats=20, random_state=42, n_jobs=-1)

with mlflow.start_run(run_name='RF Optuna FS') as run:
    for i in range(5,21):
        # select top n features
        top_n_features = (X.columns[perm_scores.importances_mean.argsort()[::-1][:i]])

        # defining the model with top 5 features
        X_new = X[top_n_features]
        
        #define the model
        model_new = RandomForestClassifier(**RF_params)
        
        n_f1 = cross_val_score(model_new, X, y, cv=skf, scoring='f1_macro', n_jobs=-1).mean()
        
        if n_f1 > RF_f1:    
            #fit the model
            model_new.fit(X_new, y)
            best_threshold_fs, threshold_f1_fs = find_best_threshold(model_new, X, y)
            mlflow.sklearn.log_model(model_new, 'model', input_example=X_new.head())
            mlflow.log_metric('f1', n_f1)
            mlflow.log_metric('best_threshold', best_threshold_fs)
            mlflow.log_metric('threshold_f1', threshold_f1_fs)
            mlflow.sklearn.log_model(model_new, artifact_path=f"model_top_{i}")

            # log tags
            mlflow.set_tags(tags={'Project': 'Project RF Optuna FS',
                                'Opimizer': 'Optuma',
                                'Model_family': 'Random forest',
                                'feature_version': 2})
            mlflow.end_run()
            print(f"Model with top {i} features outperformed original. F1: {n_f1:.4f}")
            break
    else:
        print('No smaller F1 found with less features')
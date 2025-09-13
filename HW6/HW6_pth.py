import mlflow
import json
import pandas as pd
import optuna

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Age', axis=1)
y = df['Age']

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

# loading rf params
with open('RF_params.json', 'r') as f:
    RF_params = json.load(f)
    
RF_model = RandomForestRegressor(**RF_params)

# loading et params
with open('ET_params.json', 'r') as f:
    ET_params = json.load(f)
    
ET_model = ExtraTreesRegressor(**ET_params)

# loading gb params
with open('GB_params.json', 'r') as f:
    GB_params = json.load(f)
    
GB_model = GradientBoostingRegressor(**GB_params)

# loading xgb params
with open('XGB_params.json', 'r') as f:
    XGB_params = json.load(f)
    
XGB_model = XGBRegressor(**XGB_params)

# loading LGB params
with open('LGB_params.json', 'r') as f:
    LGB_params = json.load(f)
    
LGB_model = LGBMRegressor(**LGB_params)

# loading CAT params
with open('CAT_params.json', 'r') as f:
    CAT_params = json.load(f)
    
CAT_model = CatBoostRegressor(**CAT_params)

# defing the base models for voting regressor
base_models = [('RF', RF_model), ('ET', ET_model), ('GB', GB_model), ('XGB', XGB_model), ('LGB', LGB_model), ('CAT', CAT_model)]
def objective(trial):
    # weights for model
    rf_weight = trial.suggest_float('rf_weight', 0, 1)
    et_weight = trial.suggest_float('et_weight', 0, 1)
    gb_weight = trial.suggest_float('gb_weight', 0, 1)
    xgb_weight = trial.suggest_float('xgb_weight', 0, 1)
    lgb_weight = trial.suggest_float('lgb_weight', 0, 1)
    cat_weight = trial.suggest_float('cat_weight', 0, 1)
    
    # ensure weights sum to 1
    total_weight = rf_weight + et_weight + gb_weight + xgb_weight + lgb_weight + cat_weight
    rf_weight /= total_weight
    et_weight /= total_weight
    gb_weight /= total_weight
    xgb_weight /= total_weight
    lgb_weight /= total_weight
    cat_weight /= total_weight
    
    # define the voting regressor
    vote_md = VotingRegressor(estimators=base_models, weights=[rf_weight, et_weight, gb_weight, xgb_weight, lgb_weight, cat_weight], n_jobs=-1, verbose=0)
    
    # compute the cross val score
    scores = -1* cross_val_score(vote_md, X, y, cv=skf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
    return scores

#store the best params
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, n_jobs=-1)

# get best weights
best_weights = study.best_params
w_total = best_weights['rf_weight'] + best_weights['et_weight'] + best_weights['gb_weight'] + best_weights['xgb_weight'] + best_weights['lgb_weight'] + best_weights['cat_weight']
best_weights['rf_weight'] /= w_total
best_weights['et_weight'] /= w_total
best_weights['gb_weight'] /= w_total
best_weights['xgb_weight'] /= w_total
best_weights['lgb_weight'] /= w_total
best_weights['cat_weight'] /= w_total

# define the voting regressor
vote_md = VotingRegressor(estimators=base_models, weights=[best_weights['rf_weight'], best_weights['et_weight'], best_weights['gb_weight'], best_weights['xgb_weight'], best_weights['lgb_weight'], best_weights['cat_weight']], n_jobs=-1, verbose=False)

# compute the cross val score
scores = -1* cross_val_score(vote_md, X, y, cv=skf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

# setting the experiment
mlflow.set_experiment('HW6')

with mlflow.start_run(run_name='Voting Regressor Optuna') as run:
    #define the model
    model = vote_md
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('rmse', scores)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'HW6 Voting Optuna',
                         'Model_family': 'Voting Regressor',
                         'feature_version': 1})
    mlflow.end_run()
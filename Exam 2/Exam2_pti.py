import mlflow
import json
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Rented Bike Count', axis=1)
y = df['Rented Bike Count']

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

# defing the base models for stacking regressor
base_models = [('RF', RF_model), ('ET', ET_model), ('GB', GB_model), ('XGB', XGB_model), ('LGB', LGB_model), ('CAT', CAT_model)]

# define the stacking regressor
stack_md = StackingRegressor(estimators=base_models, final_estimator=Ridge(), n_jobs=-1, cv=KFold(n_splits=5, shuffle=True, random_state=10))

# compute the cross val score
scores = -1* cross_val_score(stack_md, X, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()

# setting the experiment
mlflow.set_experiment('Exam2')

with mlflow.start_run(run_name='Stacking Regressor') as run:
    #define the model
    model = stack_md
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('rmse', scores)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'Exam 2 Stacking Regressor',
                         'Model_family': 'Stacking Regressor',
                         'feature_version': 1})
    mlflow.end_run()
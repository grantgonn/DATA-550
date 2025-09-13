import pandas as pd 
import mlflow

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

# Load the data
df = pd.read_csv('CrabAgePrediction.csv')

X = df.drop('Age', axis=1)
y = df['Age']

X = pd.get_dummies(X, columns=['Sex'], drop_first=True, dtype='int')

# define kfold

skf = KFold(n_splits=5, shuffle=True, random_state=42)

# setting mlflow
mlflow.set_experiment('grant-crab-pred')
mlflow.xgboost.autolog()

#starting with mlflow run
with mlflow.start_run(run_name='XGBoostBaseline'):
    # Create a RandomForestClassifier
    md = XGBRegressor(n_estimators=100, objective='reg:absoluteerror', learning_rate=0.1 , random_state=42)

    # Train the model using KFold
    scores = cross_val_score(md, X, y, cv=skf, scoring='neg_mean_absolute_error')
    rmse = -1*scores.mean()

    # Log the mean score to mlflow
    mlflow.log_metric('rmse', scores.mean())

import pandas as pd 
import mlflow

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score

# Load the data
df = pd.read_csv('insurance.csv')

X = df.drop('charges', axis=1)
y = df['charges']

X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True, dtype='int')

# define kfold

skf = KFold(n_splits=5, shuffle=True, random_state=42)

# setting mlflow
mlflow.set_experiment('grant-insurance-pred')
mlflow.sklearn.autolog()

#starting with mlflow run
with mlflow.start_run(run_name='extraTreesbaseline'):
    # Create a RandomForestClassifier
    md = ExtraTreesRegressor(n_estimators=100, random_state=42)

    # Train the model using KFold
    scores = cross_val_score(md, X, y, cv=skf, scoring='neg_root_mean_squared_error')
    rmse = -1*scores.mean()

    # Log the mean score to mlflow
    mlflow.log_metric('rmse', scores.mean())











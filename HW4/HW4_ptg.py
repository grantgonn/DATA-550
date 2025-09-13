import pandas as pd 
import mlflow
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import optuna

# Load the data
df = pd.read_csv('train.csv')

X = df.drop('Age', axis=1)
y = df['Age']

# define kfold
skf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial, X, y):
    params = dict(iterations=trial.suggest_int('iterations', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 10),
                    random_state=trial.suggest_categorical('random_state', [42]),
                    verbose = trial.suggest_categorical('verbose', [0]))

    # Create a CatBoostRegressor model
    md = CatBoostRegressor(**params)

    # Train the model using KFold
    scores = -1* cross_val_score(md, X, y, cv=skf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()

    return scores

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=30, n_jobs=-1)
CAT_params = study.best_params
CAT_rmse = study.best_value

mlflow.set_experiment('HW4')
#mlflow.sklearn.autolog()

# perfroming permutation importance and partial dependence
def plot_perm_imp_depen(X, y, model):
    # train model
    model_fit = model.fit(X, y)
    
    #compute permutation importance
    perm_importance = permutation_importance(model_fit, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    
    # create df for permutation importance
    perm_df = pd.DataFrame({'Feature': X.columns, 'Importance': perm_importance.importances_mean}).sort_values('Importance', ascending=False, inplace=False)
    
    #plot permutation importance
    plt.figure(figsize=(12, 8))
    perm_df.plot(kind='barh', x='Feature', y='Importance', title='Permutation Importance', legend=False)
    
    plt.savefig('CAT_importance.png')
    plt.show()
    
    #plot partial dependence
    fig, ax1 = plt.subplots(3, 3,  figsize=(20, 20))
    fig.suptitle('CAT Partial Dependence')
    PartialDependenceDisplay.from_estimator(model_fit, X, features=X.columns.to_list(), kind='both', pd_line_kw={'color':'red'}, ice_lines_kw={'color': 'steelblue', 'alpha': .1}, ax=ax1, n_jobs=-1)
    plt.savefig('CAT_partial_dependence.png')
    plt.show()
    
# running permutation importance and partial dependence
plot_perm_imp_depen(X, y, CatBoostRegressor(**CAT_params))

with mlflow.start_run(run_name='CAT Optuna') as run:
    # log feature importance and partial dependence
    mlflow.log_artifact('CAT_importance.png')
    mlflow.log_artifact('CAT_partial_dependence.png')
    
    #log best params
    mlflow.log_params(CAT_params)
    
    #define the model
    model = CatBoostRegressor(**CAT_params)
    
    #fit the model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'model', input_example=X.head())
    
    # log score
    mlflow.log_metric('rmse', CAT_rmse)
    
    # log tags
    mlflow.set_tags(tags={'Project': 'HW4 CAT Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'CAT',
                         'feature_version': 1})
    mlflow.end_run()

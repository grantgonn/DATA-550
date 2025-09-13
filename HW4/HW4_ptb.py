import pandas as pd 
import mlflow
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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
    
    plt.savefig('RF_importance.png')
    plt.show()
    
    #plot partial dependence
    fig, ax1 = plt.subplots(3, 3,  figsize=(20, 20))
    fig.suptitle('RF Partial Dependence')
    PartialDependenceDisplay.from_estimator(model_fit, X, features=X.columns.to_list(), kind='both', pd_line_kw={'color':'red'}, ice_lines_kw={'color': 'steelblue', 'alpha': .1}, ax=ax1, n_jobs=-1)
    plt.savefig('RF_partial_dependence.png')
    plt.show()
    
# running permutation importance and partial dependence
plot_perm_imp_depen(X, y, RandomForestRegressor(**RF_params))

with mlflow.start_run(run_name='RF Optuna') as run:
    # log feature importance and partial dependence
    mlflow.log_artifact('RF_importance.png')
    mlflow.log_artifact('RF_partial_dependence.png')
    
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
    mlflow.set_tags(tags={'Project': 'HW4 RF Optuna',
                         'Opimizer': 'Optuma',
                         'Model_family': 'Random forest',
                         'feature_version': 1})
    mlflow.end_run()

import pandas as pd 
import mlflow
import tensorflow as tf
import numpy as np

from sklearn.model_selection import KFold

# Load the data
df = pd.read_csv('CrabAgePrediction.csv')

X = df.drop('Age', axis=1)
y = df['Age']

X = pd.get_dummies(X, columns=['Sex'], drop_first=True, dtype='int')

# define kfold

skf = KFold(n_splits=5, shuffle=True, random_state=42)

# setting mlflow
mlflow.set_experiment('grant-crab-pred')
mlflow.sklearn.autolog()

#starting with mlflow run
with mlflow.start_run(run_name='tensorflowbaseline'):
    # Create a RandomForestClassifier
    tf_md = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    tf_md.compile(optimizer='adam', loss='MeanAbsoluteError')
    
    # cv
    metric = []
    for train_index, test_index in skf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        tf_md.fit(X_train, y_train, epochs=30, batch_size = 64, verbose=0)
        loss, rmse = tf_md.evaluate(X_test, y_test)
        metric.append(rmse)
        
    mlflow.log_metric('rmse', np.mean(metric))

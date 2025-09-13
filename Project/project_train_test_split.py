import pandas as pd 

from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Diabetes_012']) 

#save the train and test data
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)

import pandas as pd 

from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('CrabAgePrediction.csv')

df = pd.get_dummies(df, columns=['Sex'], drop_first=True, dtype='int')

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42) 

#save the train and test data
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)

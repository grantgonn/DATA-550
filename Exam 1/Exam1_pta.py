import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('SeoulBikeData.csv', encoding='latin1')
df = pd.get_dummies(df, columns=['Seasons', 'Holiday', 'Functioning Day'], drop_first=True, dtype=int)

# Plot the data
plt.figure(figsize=(8, 5))

sns.histplot(df['Rented Bike Count'], kde=True, color='blue', bins=50)
plt.title('Rented Bike Count Distribution')
plt.xlabel('Rented Bike Count')
plt.ylabel('Frequency')
plt.show()
# the distribution of the rented bike count is right-skewed, most of the bike counts are less than 500

fig, ax = plt.subplots(1, 4, figsize=(15, 5))

sns.histplot(df['Temperature(°C)'], kde=True, color='blue', bins=50, ax=ax[0])
ax[0].set_title('Temperature Distribution')
ax[0].set_xlabel('Temperature (°C)')
ax[0].set_ylabel('Frequency')

sns.histplot(df['Humidity(%)'], kde=True, color='blue', bins=50, ax=ax[1])
ax[1].set_title('Humidity Distribution')
ax[1].set_xlabel('Humidity (%)')
ax[1].set_ylabel('Frequency')

sns.histplot(df['Wind speed (m/s)'], kde=True, color='blue', bins=50, ax=ax[2])
ax[2].set_title('Wind Speed Distribution')
ax[2].set_xlabel('Wind Speed (m/s)')
ax[2].set_ylabel('Frequency')

sns.histplot(df['Dew point temperature(°C)'], kde=True, color='blue', bins=50, ax=ax[3])
ax[3].set_title('Dew Point Temperature Distribution')
ax[3].set_xlabel('Dew Point Temperature (°C)')
ax[3].set_ylabel('Frequency')
plt.show()
# from the plots, we can see temperature and humidity are normally distributed, while wind speed and dew point are slightly reight-skewed and left-skewed respectively.

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.scatterplot(x='Rainfall(mm)', y='Rented Bike Count', data=df, ax=ax[0])
ax[0].set_title('Rainfall(mm) vs Rented Bike Count')
ax[0].set_xlabel('Rainfall(mm)')
ax[0].set_ylabel('Rented Bike Count')

sns.scatterplot(x='Snowfall (cm)', y='Rented Bike Count', data=df, ax=ax[1])
ax[1].set_title('Snowfall (cm) vs Rented Bike Count')
ax[1].set_xlabel('Snowfall (cm)')
ax[1].set_ylabel('Rented Bike Count')
plt.show()
# from the plots, we can see most of the bike counts are when there is no rainfall or snowfall

plt.figure(figsize=(15, 5))
sns.lineplot(x='Date', y='Rented Bike Count', data=df, color='blue')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b is for abbreviated month (e.g., 'Jan')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Show ticks every month
plt.xticks(rotation=45)
plt.title('Time Series with Month Labels', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.tight_layout()
plt.show()
# from the plot, we can see the bike counts are higher in the summer and lower in the winter, throughout the year there are big drops in bike counts. 

# engineering new features
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = df['Date'].dt.weekday >= 5

df = df.drop('Date', axis=1)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42) 

#save the train and test data
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)
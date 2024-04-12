import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

from data_array_normalizer import data_array_normalization
from data import data_preparation


df = data_preparation()
# Group the data by 'MODEL' and 'D_REG' to count registrations per model and date
df_grouped = df.groupby(['MODEL', df['D_REG'].dt.to_period('M')]).size().reset_index(name='REGISTRATION_COUNT')

# Convert 'D_REG' back to datetime from period
df_grouped['D_REG'] = df_grouped['D_REG'].dt.to_timestamp()

# Split the data into features and target
X = df_grouped[['MODEL', 'D_REG']]
y = df_grouped['REGISTRATION_COUNT']

# Convert 'MODEL' to categorical data type
X.loc[:, 'MODEL'] = X['MODEL'].astype('category').cat.codes


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Converting 'Timestamp' columns to numeric values because 'D_REG' is the timestamp column
X_train['D_REG'] = (X_train['D_REG'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
X_test['D_REG'] = (X_test['D_REG'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

# Create the Random Forest model
rand_forest_regressor = RandomForestRegressor(random_state=42)

# Fit the model
rand_forest_regressor.fit(X_train, y_train)

y_pred_aligned, y_test, df_aligned = data_array_normalization(rand_forest_regressor, X_test, y_test, df)
# Scatter plot of actual vs predicted registration counts
sns.scatterplot(x=y_pred_aligned, y=y_test, hue=df_aligned['MODEL'], alpha=0.5, palette='tab10')
plt.xlabel('Predicted Registration Count')
plt.ylabel('Actual Registration Count')
plt.title('Random Forest Model: Actual vs Predicted Registration Counts')
plt.show()

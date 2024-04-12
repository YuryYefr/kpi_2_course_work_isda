from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from data import data_preparation
from data_array_normalizer import data_array_normalization

# Group data by model and count registrations
df = data_preparation()
df_grouped = df.groupby(['MODEL', df['D_REG'].dt.year]).size().reset_index(name='REGISTRATION_COUNT')

# Merge the registration count back to the original dataframe
df = df.merge(df_grouped, on='MODEL')

# Prepare features and target variable for regression
features = ['MAKE_YEAR', 'CAPACITY', 'OWN_WEIGHT', 'TOTAL_WEIGHT']
X = df[features]
y = df['REGISTRATION_COUNT']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the decision tree regressor
tree_regressor = DecisionTreeRegressor(random_state=42)

# Train the model
tree_regressor.fit(X_train, y_train)
y_pred_aligned, y_test, df_aligned = data_array_normalization(tree_regressor, X_test, y_test, df)
# Plotting actual vs predicted registration counts
plt.figure(figsize=(8, 6))

# Scatter plot of actual vs predicted registration counts
sns.scatterplot(x=y_pred_aligned, y=y_test, hue=df_aligned['MODEL'], alpha=0.5, palette='tab10')


# Add labels
plt.xlabel('Predicted Registration Count')
plt.ylabel('Actual Registration Count')
plt.title('Actual vs Predicted Registration Count (Decision Tree)')

# Add a line for perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

# Add a legend
plt.legend()

# Show the plot
plt.show()

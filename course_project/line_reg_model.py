import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split

from course_project.data_array_normalizer import data_array_normalization
from data import data_preparation

df = data_preparation()
# Group data by model and count registrations
df_grouped = df.groupby('MODEL').size().reset_index(name='REGISTRATION_COUNT')

# Merge the registration count back to the original dataframe
df = df.merge(df_grouped, on='MODEL')

# Prepare features and target variable for regression
features = ['MAKE_YEAR', 'CAPACITY', 'OWN_WEIGHT', 'TOTAL_WEIGHT']
X = df[features]
# Handle missing values by imputing with the mean of each column
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = df['REGISTRATION_COUNT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X, y)
y_pred_aligned, y_test, df_aligned = data_array_normalization(linear_reg, X_test, y_test, df)
# Make predictions
predicted = linear_reg.predict(X)

# Scatter plot of actual vs predicted registration counts
sns.scatterplot(x=predicted, y=y, hue=df['MODEL'], alpha=0.5, palette='tab10')

# Plotting the regression line
sns.regplot(x=predicted, y=y, scatter=False, color='red', line_kws={'label': 'Regression Line'})

# Labeling the plot
plt.xlabel('Predicted Registration Count')
plt.ylabel('Actual Registration Count')
plt.title('Actual vs Predicted Registration Count')
plt.show()

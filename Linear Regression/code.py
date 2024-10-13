
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Load the California Housing dataset
california = datasets.fetch_california_housing()

# Convert dataset to a pandas DataFrame for better visualization
data = pd.DataFrame(california.data, columns=california.feature_names)
data['target'] = california.target

# Display the first few rows of the dataset
print("First few rows of the California Housing dataset:")
print(data.head())

# ---------------- Data Preprocessing ----------------

# Log-transform the target to handle skewness
data['target'] = np.log1p(data['target'])

# Scale the features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[california.feature_names])

# Update the DataFrame with scaled features
data[california.feature_names] = scaled_features

# ---------------- Simple Linear Regression ----------------

# Use only one feature (e.g., the 'MedInc' - median income) for simple linear regression
X_simple = data[['MedInc']].values  # Selecting median income as the feature
y = data['target'].values  # Log-transformed target variable (median house value)

# Display data before splitting
plt.scatter(X_simple, y, color='blue')
plt.title('Data Before Splitting (Simple Linear Regression)')
plt.xlabel('Median Income (Standardized)')
plt.ylabel('Log Median House Value')
plt.show()

# Split the data into training (80%) and testing (20%) sets for simple linear regression
X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Create a LinearRegression model for simple linear regression
model_simple = LinearRegression()

# Train the model on the training data
model_simple.fit(X_train_simple, y_train)

# Predict the target values for the test set
y_pred_simple = model_simple.predict(X_test_simple)

# Plot the test data points and the regression line for simple linear regression
plt.scatter(X_test_simple, y_test, color='blue', label='Actual Data')
plt.plot(X_test_simple, y_pred_simple, color='red', label='Regression Line')
plt.title('Simple Linear Regression - Test Data and Regression Line')
plt.xlabel('Median Income (Standardized)')
plt.ylabel('Log Median House Value')
plt.legend()
plt.show()

# Display the actual vs predicted values as a table for simple linear regression
simple_results = pd.DataFrame({'Actual': np.expm1(y_test), 'Predicted': np.expm1(y_pred_simple)})
print("\nSimple Linear Regression - Actual vs Predicted Values:")
print(simple_results.head())

# Calculate Mean Squared Error (MSE) and R-squared for simple linear regression
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f'\nSimple Linear Regression - Mean Squared Error (MSE): {mse_simple}')
print(f'Simple Linear Regression - R-squared: {r2_simple}')

# ---------------- Multiple Linear Regression ----------------

# Use all features for multiple linear regression
X_multiple = data[california.feature_names].values  # All scaled features
y = data['target'].values  # Log-transformed target variable

# Split the data into training (80%) and testing (20%) sets for multiple linear regression
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Create a LinearRegression model for multiple linear regression
model_multiple = LinearRegression()

# Train the model on the training data
model_multiple.fit(X_train_multiple, y_train)

# Predict the target values for the test set
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Display the actual vs predicted values as a table for multiple linear regression
multiple_results = pd.DataFrame({'Actual': np.expm1(y_test), 'Predicted': np.expm1(y_pred_multiple)})
print("\nMultiple Linear Regression - Actual vs Predicted Values:")
print(multiple_results.head())

# Calculate Mean Squared Error (MSE) and R-squared for multiple linear regression
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print(f'\nMultiple Linear Regression - Mean Squared Error (MSE): {mse_multiple}')
print(f'Multiple Linear Regression - R-squared: {r2_multiple}')

# Plot actual vs predicted values for multiple linear regression
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), np.expm1(y_test), color='blue', marker='o', linestyle='', label='Actual Values')
plt.plot(range(len(y_pred_multiple)), np.expm1(y_pred_multiple), color='red', marker='x', linestyle='', label='Predicted Values')
plt.title('Multiple Linear Regression - Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

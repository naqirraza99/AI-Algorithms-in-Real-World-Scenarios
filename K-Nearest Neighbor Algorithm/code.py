# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles  # Generates a synthetic dataset of concentric circles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generating the synthetic 'circles' dataset
# - n_samples: number of samples
# - noise: standard deviation of Gaussian noise added to the data
# - factor: scaling factor between the inner and outer circle
# - random_state: ensures reproducibility
X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)

# Display basic dataset information
print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")  # Number of data points
print(f"Number of features: {X.shape[1]}")  # Number of features per data point (2D)
print(f"Class distribution: {np.bincount(y)}\n")  # Class distribution (binary classification)

# Convert dataset to a DataFrame for easier visualization and handling
# The DataFrame includes two features and the target class (0 or 1)
circles_df = pd.DataFrame(data=np.c_[X, y], columns=['Feature 1', 'Feature 2', 'target'])
circles_df['target'] = circles_df['target'].map({0: 'Class 0', 1: 'Class 1'})  # Map target values to class labels

# Display the first few rows of the dataset
display(circles_df.head())

# Visualizing the data before applying k-NN classification
# Scatter plot of the first two features, with colors representing the class labels
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.title('Circles Dataset Visualization (Before Classification)')
plt.xlabel('Feature 1')  # X-axis label
plt.ylabel('Feature 2')  # Y-axis label
colorbar = plt.colorbar(label='Classes')  # Color bar indicating the class labels
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(['Class 0', 'Class 1'])  # Map the ticks to the class names
plt.show()

# Splitting the dataset into training and testing sets
# 70% of the data is used for training, and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# k-Nearest Neighbors (k-NN) classifier
# - n_neighbors: number of neighbors to consider (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)  # Train the model on the training data

# Making predictions on the test set
y_pred = knn.predict(X_test)

# Evaluating the classifier's performance
# - accuracy_score: measures the proportion of correctly predicted instances
# - classification_report: provides precision, recall, and F1-score for each class
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print accuracy and classification report
print(f"Accuracy of k-NN: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)

# Visualizing the data after classification using the test set
# Scatter plot with predicted class labels
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Circles Dataset Classification Visualization (After Classification)')
plt.xlabel('Feature 1')  # X-axis label
plt.ylabel('Feature 2')  # Y-axis label
colorbar = plt.colorbar(label='Predicted Classes')
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(['Class 0', 'Class 1'])  # Map the ticks to predicted class names
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("../input/parkinsons/datasets_410614_786211_parkinsons.csv")

# Split features and target variable
X = data.drop(['name', 'status'], axis=1)
Y = data['status']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# Standardize the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Predictions on training data
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print("Training Accuracy:", training_accuracy)

# Predictions on testing data
Y_test_pred = model.predict(X_test)
testing_accuracy = accuracy_score(Y_test, Y_test_pred)
print("Testing Accuracy:", testing_accuracy)

# Confusion matrix and classification report for testing data
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_test_pred))
print("\nClassification Report:")
print(classification_report(Y_test, Y_test_pred))

# Example prediction
input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)
input_np = np.asarray(input_data).reshape(1, -1)
standard_input = scaler.transform(input_np)
prediction = model.predict(standard_input)
if prediction[0] == 1:
    print("The patient has Parkinson's disease.")
else:
    print("The person is healthy.")

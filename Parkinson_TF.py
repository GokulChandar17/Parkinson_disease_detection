import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, Y_train, verbose=0)
print("Training Accuracy:", train_accuracy)

test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Testing Accuracy:", test_accuracy)

# Predictions
Y_train_pred = model.predict(X_train)
Y_train_pred_binary = (Y_train_pred > 0.5).astype(int)
Y_test_pred = model.predict(X_test)
Y_test_pred_binary = (Y_test_pred > 0.5).astype(int)

# Confusion matrix and classification report for testing data
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_test_pred_binary))
print("\nClassification Report:")
print(classification_report(Y_test, Y_test_pred_binary))

# Example prediction
input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)
input_np = np.asarray(input_data).reshape(1, -1)
standard_input = scaler.transform(input_np)
prediction = model.predict(standard_input)
if prediction[0][0] > 0.5:
    print("The patient has Parkinson's disease.")
else:
    print("The person is healthy.")

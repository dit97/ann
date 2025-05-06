#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Replace this line:
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# df = pd.read_csv(url, names=column_names)
# With:
# local_path = "pima-indians-diabetes.data.csv"  # Ensure this path is correct
# df = pd.read_csv(local_path, names=column_names)



# Import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the Pima Indians Diabetes dataset from a CSV URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# Split dataset into features (X) and labels (y)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a logistic regression model using a single Dense layer with sigmoid activation
logistic_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),          # 8 input features
    tf.keras.layers.Dense(1, activation='sigmoid')      # Output probability (binary classification)
])

# Compile the model with Adam optimizer and binary crossentropy loss
logistic_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Train the model on the training data
logistic_model.fit(X_train, y_train, epochs=100, verbose=0)  # Train silently

# Evaluate the model on test data
loss, acc = logistic_model.evaluate(X_test, y_test, verbose=0)
print(f"Logistic Regression Accuracy: {acc:.4f}")

# Predict probabilities and convert to class labels (threshold = 0.5)
y_pred_proba = logistic_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Print classification report
print(classification_report(y_test, y_pred))


# In[ ]:





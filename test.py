# test.py

import numpy as np
from tensorflow.keras.models import model_from_json

# Load the dataset (or a portion of it for testing)
dataset = np.loadtxt('diabetes.csv', delimiter=',')
X_test = dataset[:, 0:8]
y_test = dataset[:, 8]

# --- Load the saved model ---
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights("model.weights.h5")
    print("\nLoaded model from disk")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Make predictions ---
# Predict probabilities
probabilities = model.predict(X_test)
# Convert probabilities to binary predictions (0 or 1)
predictions = (probabilities > 0.5).astype(int).flatten()

# --- Show some example predictions ---
print("\n--- Example Predictions ---")
for i in range(10): # Show first 10 examples
    print(f'Input: {X_test[i]}')
    print(f'Predicted: {predictions[i]}, Expected: {int(y_test[i])}')
    print('---')
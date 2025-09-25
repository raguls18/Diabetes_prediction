# train.py (using Scikit-learn for deployment)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load data with Pandas
print("Loading dataset...")
df = pd.read_csv('diabetes_health.csv')

# Simplify the target to a binary problem: 0 = no diabetes, 1 = diabetes.
df['Diabetes_012'] = df['Diabetes_012'].apply(lambda x: 0 if x == 0 else 1)

# Separate features (X) and target (y)
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. Define and Train the Scikit-learn Model ---
print("Training RandomForest model... (This will be much faster)")
# n_estimators is the number of trees in the forest. n_jobs=-1 uses all available CPU cores.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training finished.")

# --- 3. Evaluate the model's performance ---
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

# --- 4. Save the new, lighter model to a single file ---
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
print("Saved Scikit-learn model to model.pkl")
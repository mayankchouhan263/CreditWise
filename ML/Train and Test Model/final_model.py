import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv("ML/Data/Processed_loan_approval_data.csv")

# Drop target — only once, no extra columns
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

print("Features:", X.shape[1])  # must print 27

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dt_model = DecisionTreeClassifier(max_depth=15, min_samples_leaf=50, random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

# Get directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(dt_model, os.path.join(MODEL_DIR, "loan_model.pkl"))
print("Model saved to:", MODEL_DIR)

# Evaluation
# print("Decision Tree Model")
# print("Precision: ", precision_score(y_test, y_pred))
# print("Recall: ", recall_score(y_test, y_pred))
# print("F1 score: ", f1_score(y_test, y_pred))
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("CM: ")
# print(confusion_matrix(y_test, y_pred))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
)

# 1. SETUP AND DATA LOADING
# Creating a results directory for organized output
os.makedirs("results", exist_ok=True)

print("--- Loading Data and Model ---")
X_test = pd.read_csv("datasets/test_features.csv")
y_test = pd.read_csv("datasets/test_target.csv").squeeze()

# Loading the Random Forest model (as it usually performs best for this task)
# Note: RF doesn't require scaling, so we use X_test directly as per your train_model.py
model = joblib.load("models/random_forest.pkl")

# 2. GENERATE PREDICTIONS
print("--- Generating Predictions ---")
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

# 3. ERROR ANALYSIS (Metrics)
print("\n--- Error Analysis Metrics ---")
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred)
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 4. LOGGING PREDICTIONS TO CSV
print("\n--- Logging Predictions ---")
results_df = X_test.copy()
results_df['Actual_Default'] = y_test.values
results_df['Predicted_Default'] = y_pred
results_df['Probability_of_Default'] = y_probs

results_df.to_csv("results/final_predictions.csv", index=False)
print("✅ Predictions saved to 'results/final_predictions.csv'")

# 5. VISUALIZATIONS
print("--- Generating Plots ---")

# Plot 1: Confusion Matrix

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: Loan Default Prediction")
plt.savefig("results/confusion_matrix.png")
plt.show()

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("results/roc_curve.png")
plt.show()

print("\n✅ All results and plots saved in the 'results/' folder.")
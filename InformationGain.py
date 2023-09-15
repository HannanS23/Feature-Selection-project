import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# Insert file (of extracted features)
file_path = ''

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Assuming you have your dataset loaded into a DataFrame called 'data'
X = data.drop("Grade", axis=1)  # Features
y = data["Grade"]  # Target variable

# Calculate Information Gain scores for each feature
info_gain_scores = mutual_info_classif(X, y)

# Sort the features based on Information Gain scores in descending order
sorted_features_indices = info_gain_scores.argsort()[::-1]

# Select the top k features based on Information Gain scores (you can adjust the value of k as needed)
k = 8
selected_feature_indices = sorted_features_indices[:k]

# Get the selected feature names
selected_feature_names = X.columns[selected_feature_indices]

print(selected_feature_names)

# Create a DataFrame with the selected features
X_selected = X[selected_feature_names]

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Perform cross-validation and get predicted labels for each fold
# 'cv' specifies the number of folds for cross-validation (e.g., cv=5 for 5-fold cross-validation)
cv_predictions = cross_val_predict(svm_classifier, X_selected, y, cv=5)

# Calculate evaluation metrics for the cross-validated predictions
accuracy = accuracy_score(y, cv_predictions)
precision = precision_score(y, cv_predictions)
recall = recall_score(y, cv_predictions)
f1 = f1_score(y, cv_predictions)

# Since specificity is not available as a direct metric in scikit-learn, we can calculate it manually
tn, fp, fn, tp = confusion_matrix(y, cv_predictions).ravel()
specificity = tn / (tn + fp)

# Get predicted probabilities for calculating AUC-ROC
svm_classifier_prob = SVC(kernel='linear', C=1.0, probability=True)
cv_prob_predictions = cross_val_predict(svm_classifier_prob, X_selected, y, cv=5, method='predict_proba')
auc_roc = roc_auc_score(y, cv_prob_predictions[:, 1])

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)


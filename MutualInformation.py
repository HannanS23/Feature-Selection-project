import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer)

# Insert file (of extracted features)
file_path = ''

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

X = data.drop("Grade", axis=1)  # Features
y = data["Grade"]  # Target variable

# Compute mutual information scores for each feature
mutual_info_scores = mutual_info_classif(X, y)

# Create a DataFrame to store feature names and their corresponding mutual information scores
feature_scores_df = pd.DataFrame({'Feature': X.columns, 'Mutual_Info_Score': mutual_info_scores})

# Sort the features based on their mutual information scores in descending order
sorted_features = feature_scores_df.sort_values(by='Mutual_Info_Score', ascending=False)

# Print the sorted features and their corresponding mutual information scores
#print(sorted_features)

# Select the top 10 features
top_10_features = sorted_features.head(10)

# Print the top 10 features and their corresponding mutual information scores
#print(top_10_features)

# Extract the feature names from the top 10 rows
selected_feature_names = top_10_features['Feature']

# Create a new DataFrame containing only the top 10 features
X_selected = X[selected_feature_names]

# Print X_selected
#print(X_selected)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Perform cross-validation and get predicted labels for each fold
# 'cv' specifies the number of folds for cross-validation (e.g., cv=5 for 5-fold cross-validation)
cv_predictions = cross_val_predict(svm_classifier, X_selected, y, cv=3)

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
cv_prob_predictions = cross_val_predict(svm_classifier_prob, X_selected, y, cv=3, method='predict_proba')
auc_roc = roc_auc_score(y, cv_prob_predictions[:, 1])

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)
# Predict tumor grades using the Lasso model and selected features
predicted_grades = svm_predicted_grades




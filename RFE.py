import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

file_path = 'C:/Users/shaha/OneDrive/MSc Data Science & Business Analytics/PROJ518/RFE/Code3/Feature_Extraction_New.xlsx'

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Assuming you have your dataset loaded into a DataFrame called 'data'
X = data.drop("Grade", axis=1)  # Features
y = data["Grade"]  # Target variable

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Apply Recursive Feature Elimination (RFE) for feature selection
num_features_to_select = 8 # Number of features to select (adjust as needed)
rfe = RFE(estimator=svm_classifier, n_features_to_select=num_features_to_select)
X_rfe_selected = rfe.fit_transform(X, y)

print(X_rfe_selected)


# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Perform cross-validation and get predicted labels for each fold
# 'cv' specifies the number of folds for cross-validation (e.g., cv=5 for 5-fold cross-validation)
cv_predictions = cross_val_predict(svm_classifier, X_rfe_selected, y, cv=3)

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
cv_prob_predictions = cross_val_predict(svm_classifier_prob, X_rfe_selected, y, cv=3, method='predict_proba')
auc_roc = roc_auc_score(y, cv_prob_predictions[:, 1])

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)

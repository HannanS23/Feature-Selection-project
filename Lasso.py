import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
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

# Read the Excel file (of extracted features) into a DataFrame
data = pd.read_excel(file_path)


X = data.drop("Grade", axis=1)  # Features
y = data["Grade"]  # Target variable

# Standardize the features (important for regularization methods like Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Lasso feature selection
alpha = 0.01 # Regularization strength (adjust as needed)
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_scaled, y)

# Get the coefficients of the features
feature_coeffs = lasso_model.coef_

# Get the selected feature names (where coefficient is not zero)
selected_feature_names = X.columns[feature_coeffs != 0]

# Print the selected feature names
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

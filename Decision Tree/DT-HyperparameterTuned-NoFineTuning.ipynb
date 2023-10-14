import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Dataset features and target
features = ['AGE', 'SEX', 'eGFR_slope', 'eGFR_mean']
X_test = TEST_df[features]
y_test = TEST_df['Kidney_Failure']

# # Train the model on 15% of the data (Fine tuning)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=seed_value, stratify=y)
# clf.fit(X_train, y_train)

# Evaluate on the held-out test set
clf = best_clf

y_pred = clf.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, y_pred))
print("Final Classification Report:")
print(classification_report(y_test, y_pred))
print("Final ROC-AUC Score:")
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# Compute the statistics
true_positives = np.sum((y_pred == 1) & (y_test == 1))
true_negatives = np.sum((y_pred == 0) & (y_test == 0))
false_positives = np.sum((y_pred == 1) & (y_test == 0))
false_negatives = np.sum((y_pred == 0) & (y_test == 1))

# Print individual metrics
tpr = true_positives / (true_positives + false_negatives)
tnr = true_negatives / (true_negatives + false_positives)
fpr = false_positives / (false_positives + true_negatives)
fnr = false_negatives / (false_negatives + true_positives)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = 2 * precision * recall / (precision + recall)
specificity = true_negatives / (true_negatives + false_positives)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

print("TPR:", tpr, "TNR:", tnr, "FPR:", fpr, "FNR:", fnr,
      '\n' "Accuracy", accuracy, "Precision:", precision, "Recall:", recall, "F1 Score:", f1_score, "Specificity:", specificity, "ROC-AUC:", roc_auc)

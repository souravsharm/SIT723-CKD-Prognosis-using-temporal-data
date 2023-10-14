import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Sample dataset features and target
features = ['AGE', 'SEX', 'eGFR_slope', 'eGFR_mean']
X = df_merged[features]
y = df_merged['Kidney_Failure']

# Initialize a Decision Tree model
clf = RandomForestClassifier(random_state=seed_value) # 

param_grid = {
    'n_estimators': [30, 50],
    'max_depth': [5, 10],
    'min_samples_split': [4, 5],
    'min_samples_leaf': [2, 3],
    'max_features': ['auto']
}                                                     # 

# Initialize GridSearch
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
                           scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit the GridSearch model
grid_search.fit(X, y)

# Extract the best estimator
best_clf = grid_search.best_estimator_

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model using the best estimator
    best_clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = best_clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1]))

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
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])

    print(accuracy, precision, recall, f1_score, specificity, roc_auc)

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 10 folds
print(df_results.mean())

# Print out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model on the full Australian dataset
best_clf = RandomForestClassifier(**best_params, random_state=seed_value) # 
best_clf.fit(X, y)

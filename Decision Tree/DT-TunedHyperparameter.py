import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Sample dataset features and target
features = ['AGE', 'SEX', 'eGFR_slope', 'eGFR_mean']
X = df_merged[features]
y = df_merged['Kidney_Failure']

# Initialize accumulators for confusion matrix elements
total_true_positives = 0
total_true_negatives = 0
total_false_positives = 0
total_false_negatives = 0

# Initialize a Decision Tree model
clf = DecisionTreeClassifier(random_state=seed_value) # 

param_grid = {
    #'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    #'max_features': ['auto', 'sqrt', 'log2', None],
    #'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'class_weight': ['balanced', None]
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

    # Accumulate confusion matrix values
    total_true_positives += true_positives
    total_true_negatives += true_negatives
    total_false_positives += false_positives
    total_false_negatives += false_negatives

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

# Compute mean confusion matrix values
mean_true_positives = total_true_positives / n_splits
mean_true_negatives = total_true_negatives / n_splits
mean_false_positives = total_false_positives / n_splits
mean_false_negatives = total_false_negatives / n_splits

# Construct the mean confusion matrix
mean_confusion_matrix = np.array([[mean_true_negatives, mean_false_positives],
                                 [mean_false_negatives, mean_true_positives]])

# Plot the mean confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=['No Failure', 'Kidney Failure'])
cm_display.plot(cmap='Blues')  # Using the 'Blues' colormap
plt.title('Mean Confusion Matrix')
plt.show()

# Print the average of each statistic over the 5 folds
print(df_results.mean())

# Print out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model on the full Australian dataset
best_clf = DecisionTreeClassifier(**best_params, random_state=seed_value) # 
best_clf.fit(X, y)v

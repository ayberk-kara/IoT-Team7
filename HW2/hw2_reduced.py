# necessary libraries
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, precision_recall_fscore_support

# file paths
train_path = './HW2Data/train/'
test_path = './HW2Data/test/'

X_train_file = train_path + 'X_train.txt'
y_train_file = train_path + 'y_train.txt'
X_test_file = test_path + 'X_test.txt'
y_test_file = test_path + 'y_test.txt'

# loading the data
X_train = np.loadtxt(X_train_file)
y_train = np.loadtxt(y_train_file)
X_test = np.loadtxt(X_test_file)
y_test = np.loadtxt(y_test_file)

# normalize X_train for Chi-Squared Test (it needs non-negative values)
X_train_normalized = np.abs(X_train)

# apply chi-squared Test, select top 50 features
k_best = SelectKBest(chi2, k=50)
X_train_k_best = k_best.fit_transform(X_train_normalized, y_train)

# get scores and visualize feature importance
feature_scores = k_best.scores_  # scores for each feature
feature_names = np.arange(X_train.shape[1])  # just using feature indices

# visualize Chi-Squared Scores
plt.figure(figsize=(12, 8))
plt.barh(feature_names, feature_scores, color='lightcoral')
plt.title('Chi-Squared Test Scores for Features')
plt.xlabel('Chi-Squared Score')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.show()

# selected features list (top 50)
selected_features = k_best.get_support(indices=True)
print("Selected feature indices:", selected_features)

# filter data to include only selected features
X_train_filtered = X_train[:, selected_features]
X_test_filtered = X_test[:, selected_features]

# function to find the best k value using cross-validation
def find_best_k(X_train, y_train, k_range):
    best_score = 0
    best_k = 1

    for k in k_range:
        # KNN setup with current k
        knn = KNeighborsClassifier(n_neighbors=k)

        # cross-validation (5 folds)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')

        # mean accuracy
        mean_score = np.mean(scores)

        # update best k if better score found
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k, best_score

# find the best k value
k_range = range(1, 101)
best_k, best_score = find_best_k(X_train_filtered, y_train, k_range)
print(f"Best k value: {best_k}, Cross-validation accuracy: {best_score:.4f}")

# train KNN with the best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_filtered, y_train)
y_pred = knn_best.predict(X_test_filtered)

# show classification report
print(classification_report(y_test, y_pred))

# path for per-user combined data
per_user_folder = './HW2Data/per_user_combined/'

# function for KNN prediction optimized
def knn_predict_optimized(X_train, y_train, X_test, k=5):
    predictions = []

    for test_row in X_test:
        # calculate Euclidean distances
        distances = np.linalg.norm(X_train - test_row, axis=1)

        # pair distances with labels
        distances_with_labels = list(zip(distances, y_train))

        # sort by distance
        distances_with_labels.sort(key=lambda x: x[0])

        # adjust k if needed
        k_adjusted = min(k, len(distances_with_labels))
        nearest_neighbors = [label for _, label in distances_with_labels[:k_adjusted]]

        if nearest_neighbors:  # if we have neighbors
            prediction = Counter(nearest_neighbors).most_common(1)[0][0]
            predictions.append(prediction)
        else:
            predictions.append(None)  # no neighbors case

    return predictions

# evaluation per user
unique_users = [folder for folder in os.listdir(per_user_folder) if os.path.isdir(os.path.join(per_user_folder, folder))]
overall_precision = []
overall_recall = []
overall_f1 = []

for user_folder in unique_users:
    user_path = os.path.join(per_user_folder, user_folder)

    # load user's training and test data
    X_train_user = np.loadtxt(user_path + '/X_train_user.txt')
    y_train_user = np.loadtxt(user_path + '/y_train_user.txt')
    X_test_user = np.loadtxt(user_path + '/X_test_user.txt')
    y_test_user = np.loadtxt(user_path + '/y_test_user.txt')

    # predict using KNN for this user
    y_pred_user = knn_predict_optimized(X_train_user, y_train_user, X_test_user, k=5)

    # calculate precision, recall, F1 for this user
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_user, y_pred_user, average='weighted', zero_division=0)

    # store the results
    overall_precision.append(precision)
    overall_recall.append(recall)
    overall_f1.append(f1)

    # print report for each user
    print(f'User {user_folder} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')

# plot F1 scores for each user
plt.figure(figsize=(10, 6))
sns.barplot(x=[user_folder for user_folder in unique_users], y=overall_f1)
plt.title('F1 Score per User')
plt.xlabel('User')
plt.ylabel('F1 Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

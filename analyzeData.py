from preprocessing import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.linear_model import LogisticRegression

dataToAnalyze, target = preProcessData()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataToAnalyze, target, test_size=0.3, random_state=111)

# Handling imbalanced data
smote = SMOTE(sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Normalize data
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Apply TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=4)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)

print("Truncated SVD Complete")

classifiers = {
    "Gaussian Naive Bayes": GaussianNB(),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Linear Regression": LogisticRegression(solver='liblinear', multi_class = 'ovr')
}

print("Classifiers Established")

# Evaluate classifiers after LDA
accuracy = {}
rmse = {}
snr = {}

for name, clf in classifiers.items():
    clf.fit(X_train_svd, y_resampled)
    y_pred = clf.predict(X_test_svd)
    accuracy[name] = accuracy_score(y_test, y_pred)
    rmse[name] = root_mean_squared_error(y_test, y_pred)
    snr[name] = np.mean(y_pred) / np.std(y_pred)

# Display accuracy after LDA
print("Classification Metrics:")
for name, acc in accuracy.items():
    print(f"{name} Accuracy: {acc:.2f}")
    print(f"{name} RMSE: {rmse[name]:.2f}")
    print(f"{name} SNR: {snr[name]:.2f}")

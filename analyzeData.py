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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, root_mean_squared_error
from sklearn.linear_model import LogisticRegression

dataToAnalyze, target = preProcessData()

# Handling imbalanced data
smote = SMOTE(sampling_strategy='auto')
dataToAnalyze, target = smote.fit_resample(dataToAnalyze, target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataToAnalyze, target, test_size=0.3, random_state=111)

# Normalize data
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=6)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)

print("Truncated SVD Complete")

classifiers = {
    "Gaussian Naive Bayes": GaussianNB(),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(solver='liblinear', multi_class='ovr')
}

print("Classifiers Established")

# Evaluate classifiers
accuracy = {}
precision = {}
sensitivity = {}
f1 = {}
specificity = {}
rmse = {}
snr = {}

for name, clf in classifiers.items():
    clf.fit(X_train_svd, y_train)
    y_pred = clf.predict(X_test_svd)

    # start of data visualization code
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)

    disp.plot()
    plt.title("confusion matrix for " + name)
    plt.show()


    plt.hist(y_pred)
    plt.title("histrogram for " + name)
    plt.show()

    corr_matrices = {}
    corr_matrix = np.corrcoef(y_test, y_pred)
    corr_matrices[name] = corr_matrix

    # Display correlation matrices
    print("Correlation Matrices:")
    for name, corr_matrix in corr_matrices.items():
        print(f"{name}:")
        print(corr_matrix)

    # end of visualization code

    accuracy[name] = accuracy_score(y_test, y_pred)
    precision[name] = precision_score(y_pred, y_test)
    sensitivity[name] = recall_score(y_pred, y_test)
    f1[name] = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity[name] = tn / (tn + fp)
    rmse[name] = root_mean_squared_error(y_test, y_pred)
    snr[name] = np.mean(y_pred) / np.std(y_pred)

# code for displaying bar graph
plt.bar(classifiers.keys(), accuracy.values(), color='skyblue')
plt.title("accuracies for all classifiers")
plt.show()

# Display classification metrics
print("Classification Metrics:")
for name, acc in accuracy.items():
    print(f"{name} Accuracy: {acc:.2f}")
    print(f"{name} Precision: {precision[name]:.2f}")
    print(f"{name} Sensitivity: {sensitivity[name]:.2f}")
    print(f"{name} F1 Score: {f1[name]:.2f}")
    print(f"{name} Specificity: {specificity[name]:.2f}")
    print(f"{name} RMSE: {rmse[name]:.2f}")
    print(f"{name} SNR: {snr[name]:.2f}")


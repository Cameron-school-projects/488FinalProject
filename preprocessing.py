import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#placeholder for making api call to kaggle to update dataset
def updateData():
    updatedData = "temp"
    return updatedData


def preProcessData():
    df = pd.read_csv("fraud test.csv")
    #Drop unneeded attributes
    sns.countplot(x = 'is_fraud', data = df)
    df['is_fraud'].value_counts()
    X = df.drop(columns=['is_fraud', 'dob', 'first', 'last', 'trans_num', 'cc_num'])
    y = df['is_fraud']
    # Split features into numerical and non-numerical
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # set up our pre-processing
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create the ColumnTransformer to preprocess each column type
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Preprocess the data with the set up processors
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed,y
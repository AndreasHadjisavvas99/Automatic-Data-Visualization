import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def preprocess_data(df, apply_percentile_cutoff=True, apply_scaling=True, impute_missing=True, apply_one_hot_encoding=True):
    y = df['Plot_Type']  # Labels
    df = df.rename(columns={'Plot_Type':'Label'})

    X_categorical = pd.DataFrame()
    X_numerical = pd.DataFrame()
    X_boolean = pd.DataFrame()
    Y = df["Label"].values
    df = df.drop(labels = ['Label'], axis=1)

    # Dictionary to store column types
    column_types = {}
    numerical_means = {}
    categorical_modes = {}

    # splitting data to categorical and numerical for later purposes
    for column in df.columns:
        if df[column].dtype == 'object':
            X_categorical[column] = df[column]
            column_types[column] = 'categorical'
        elif df[column].dtype == 'bool':
            X_boolean[column] = df[column].astype(int)
            column_types[column] = 'boolean'
        else:
            X_numerical[column] = df[column]
            column_types[column] = 'numerical'

    # 1) Set numerical features above the 99th percentile or below the 1st percentile to those respective cut-offs
    if apply_percentile_cutoff:
        percentile_1 = X_numerical.quantile(0.01)
        percentile_99 = X_numerical.quantile(0.99)
        for column in X_numerical.columns:
            X_numerical[column] = np.where(X_numerical[column] > percentile_99[column], percentile_99[column], X_numerical[column])
            X_numerical[column] = np.where(X_numerical[column] < percentile_1[column], percentile_1[column], X_numerical[column])

    # 2) Remove the mean of numeric fields and scale to unit variance
    if apply_scaling:
        scaler = StandardScaler()
        numeric_data = X_numerical.select_dtypes(include=['float64', 'int64'])
        scaler.fit(numeric_data)
        scaled_numeric_data = scaler.transform(numeric_data)
        X_numerical[numeric_data.columns] = scaled_numeric_data

    #################### FLAGGING REPLACE WITH INF ########################
    # 3a) Impute missing categorical values using the mode of non missing values
    if impute_missing:
        for column in X_categorical.columns:
            #if X_categorical[column].dtype == 'object':
            mode_val = X_categorical[column].mode()[0]
            categorical_modes[column] = mode_val
            X_categorical[column].fillna(mode_val, inplace=True)

        # 3b) Impute missing numerical values with mean of non-missing values
        for column in X_numerical.columns:
            #if pd.api.types.is_numeric_dtype(X_numerical[column].dtype):
            mean_val = X_numerical[column].mean()
            median_val = X_numerical[column].median()
            numerical_means[column] = mean_val
            #X_numerical[column].fillna(np.inf, inplace=True)
            X_numerical[column].fillna(mean_val, inplace=True)


    # 4) One-hot encoding to categorical features
    if apply_one_hot_encoding:
        one_hot_encoder = OneHotEncoder()
        X_categorical_encoded = one_hot_encoder.fit_transform(X_categorical).toarray()
        encoded_feature_names = one_hot_encoder.get_feature_names_out(X_categorical.columns)
        X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names)
        X = pd.concat([X_numerical, X_categorical_encoded_df, X_boolean], axis=1)
        print("Known categories:", one_hot_encoder.categories_)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=100, stratify=Y)
    # One-hot encode the labels
    label_encoder = LabelEncoder()
    train_integer_encoded = label_encoder.fit_transform(y_train)
    test_integer_encoded = label_encoder.fit_transform(y_test)
    y_train_onehot = to_categorical(train_integer_encoded)
    y_test_onehot = to_categorical(test_integer_encoded)

    # Save preprocessing components
    preprocessing_info = {
        'column_types': column_types,
        'percentile_1': percentile_1,
        'percentile_99': percentile_99,
        'scaler': scaler,
        'categorical_modes': categorical_modes,
        'numerical_means': numerical_means,
        'one_hot_encoder': one_hot_encoder
    }

    return X_test, y_test_onehot, X_train, y_train_onehot, preprocessing_info

def preprocessing_one(df,preprocessing_info):
    column_types = preprocessing_info['column_types']
    percentile_1 = preprocessing_info['percentile_1']
    percentile_99 = preprocessing_info['percentile_99']
    scaler = preprocessing_info['scaler']
    categorical_modes = preprocessing_info['categorical_modes']
    numerical_means = preprocessing_info['numerical_means']
    one_hot_encoder = preprocessing_info['one_hot_encoder']

    y = df['Plot_Type']
    df = df.rename(columns={'Plot_Type':'Label'})

    X_categorical = pd.DataFrame()
    X_numerical = pd.DataFrame()
    X_boolean = pd.DataFrame()
    Y = df["Label"].values
    df = df.drop(labels = ['Label'], axis=1)

    for column, ctype in column_types.items():
        if ctype == 'categorical':
            X_categorical[column] = df[column]
        elif ctype == 'boolean':
            X_boolean[column] = df[column].astype(int)
        elif ctype == 'numerical':
            X_numerical[column] = df[column]

    # 1) Set numerical features above the 99th percentile or below the 1st percentile to those respective cut-offs
    percentile_1 = preprocessing_info['percentile_1']
    percentile_99 = preprocessing_info['percentile_99']
    for column in X_numerical.columns:
        X_numerical[column] = np.where(X_numerical[column] > percentile_99[column], percentile_99[column], X_numerical[column])
        X_numerical[column] = np.where(X_numerical[column] < percentile_1[column], percentile_1[column], X_numerical[column])

    # 2) Remove the mean of numeric fields and scale to unit variance
    X_numerical = pd.DataFrame(scaler.transform(X_numerical), columns=X_numerical.columns)

    # 3a) Impute missing categorical values using the mode of non missing values
    for column in X_categorical.columns:        
        X_categorical[column].fillna(categorical_modes[column], inplace=True)

    # 3b) Impute missing numerical values with mean of non-missing values
    for column in X_numerical.columns:
        X_numerical[column].fillna(numerical_means[column], inplace=True)
    
    # 4) One-hot encoding to categorical features
    print("aaaaaaaaaaaaaaaaaaaaaaa")
    print(X_categorical)
    print("aaaaaaaaaaaaaaaaaaaaaaa")
    X_categorical_encoded = one_hot_encoder.transform(X_categorical).toarray()
    encoded_feature_names = one_hot_encoder.get_feature_names_out(X_categorical.columns)
    X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names)

    X = pd.concat([X_numerical, X_categorical_encoded_df, X_boolean], axis=1)

    return X

def preprocess_for_ag(df, apply_percentile_cutoff=True, apply_scaling=True, impute_missing=True, apply_one_hot_encoding=True):
    X_categorical = pd.DataFrame()
    X_numerical = pd.DataFrame()
    X_boolean = pd.DataFrame()

    # Splitting data to categorical and numerical for later purposes
    for column in df.columns:
        if df[column].dtype == 'object':
            X_categorical[column] = df[column]
        elif df[column].dtype == 'bool':
            X_boolean[column] = df[column].astype(int)
        else:
            X_numerical[column] = df[column]

    # 1) Set numerical features above the 99th percentile or below the 1st percentile to those respective cut-offs
    if apply_percentile_cutoff:
        print("Know applying percentile cutoffs")
        percentile_1 = X_numerical.quantile(0.01)
        percentile_99 = X_numerical.quantile(0.99)
        for column in X_numerical.columns:
            X_numerical[column] = np.where(X_numerical[column] > percentile_99[column], percentile_99[column], X_numerical[column])
            X_numerical[column] = np.where(X_numerical[column] < percentile_1[column], percentile_1[column], X_numerical[column])

    # 2) Remove the mean of numeric fields and scale to unit variance
    if apply_scaling:
        print("Know Scaling data")
        scaler = StandardScaler()
        numeric_data = X_numerical.select_dtypes(include=['float64', 'int64'])
        X_numerical[numeric_data.columns] = scaler.fit_transform(numeric_data)

    # 3a) Impute missing categorical values using the mode of non-missing values
    if impute_missing:
        print("Know applying imputation to missing data")
        for column in X_categorical.columns:
            mode_val = X_categorical[column].mode()[0]
            X_categorical[column].fillna(mode_val, inplace=True)

        # 3b) Impute missing numerical values with the mean of non-missing values
        for column in X_numerical.columns:
            mean_val = X_numerical[column].mean()
            X_numerical[column].fillna(mean_val, inplace=True)

    # 4) One-hot encoding to categorical features
    if apply_one_hot_encoding:
        print("Know One-Hot Encoding")
        one_hot_encoder = OneHotEncoder()
        X_categorical_encoded = one_hot_encoder.fit_transform(X_categorical).toarray()
        encoded_feature_names = one_hot_encoder.get_feature_names_out(X_categorical.columns)
        X_categorical = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names)

    # Concatenate all the processed features
    X = pd.concat([X_numerical, X_categorical, X_boolean], axis=1)
    
    return X
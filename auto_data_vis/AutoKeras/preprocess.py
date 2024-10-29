import numpy as np
import pandas as pd
import autokeras as ak
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    X_categorical = pd.DataFrame()
    X_numerical = pd.DataFrame()
    Y = df["Label"].values
    df = df.drop(labels = ['Label'], axis=1)

    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype == 'bool':
            X_categorical[column] = df[column]
        else:
            X_numerical[column] = df[column]

    # 1) Set numerical features above the 99th percentile or below the 1st percentile to those respective cut-offs
    percentile_1 = X_numerical.quantile(0.01)
    percentile_99 = X_numerical.quantile(0.99)
    for column in X_numerical.columns: 
        X_numerical[column] = np.where(X_numerical[column] > percentile_99[column], percentile_99[column], X_numerical[column])
        X_numerical[column] = np.where(X_numerical[column] < percentile_1[column], percentile_1[column], X_numerical[column])

    # 2a) Impute missing categorical values using the mode of non missing values
    for column in X_categorical.columns:
        #if X_categorical[column].dtype == 'object':  
        mode_val = X_categorical[column].mode()[0]  
        X_categorical[column].fillna(mode_val, inplace=True)

    # 2b) Impute missing numerical values with mean of non-missing values
    for column in X_numerical.columns:
        #if pd.api.types.is_numeric_dtype(X_numerical[column].dtype): 
        mean_val = X_numerical[column].mean()
        median_val = X_numerical[column].median()
        X_numerical[column].fillna(median_val, inplace=True)

    # 3) Remove the mean of numeric fields and scale to unit variance
    scaler = StandardScaler()
    numeric_data = X_numerical.select_dtypes(include=['float64', 'int64'])
    scaler.fit(numeric_data)
    scaled_numeric_data = scaler.transform(numeric_data)
    X_numerical[numeric_data.columns] = scaled_numeric_data

    # 4) One-hot encoding to categorical features
    one_hot_encoder = OneHotEncoder()
    X_categorical_encoded = one_hot_encoder.fit_transform(X_categorical).toarray()
    encoded_feature_names = one_hot_encoder.get_feature_names_out(X_categorical.columns)
    X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names)
    X = pd.concat([X_numerical, X_categorical_encoded_df], axis=1)

    return X, Y

# Load data
df = pd.read_csv('../features1/dtv_c3.csv')
df = df.rename(columns={'Plot_Type':'Label'})


# Preprocess data
features, label = preprocess_data(df)

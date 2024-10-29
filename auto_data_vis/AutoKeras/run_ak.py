from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import autokeras as ak

def preprocess_data(df, apply_percentile_cutoff=True, apply_scaling=True, impute_missing=True, apply_one_hot_encoding=True):
    X_categorical = pd.DataFrame()
    X_numerical = pd.DataFrame()
    X_boolean = pd.DataFrame()

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

# Load your dataset
df = pd.read_csv('../features1/c2_sp.csv')
df = df.rename(columns={'Plot_Type':'Label'})

# Separate the target variable
Y = df["Label"].values
df = df.drop(labels=['Label'], axis=1)

# Preprocess the data with your desired steps
X = preprocess_data(df, apply_percentile_cutoff=True, apply_scaling=True, impute_missing=False, apply_one_hot_encoding=True)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=100)

#early_stopping = EarlyStopping(
#    monitor="val_accuracy",
#    min_delta=0,
#    patience=30,
#    verbose=1,
#    mode="auto",
#    #baseline=None,
#    restore_best_weights=False,
#    start_from_epoch=250,
#)

clf = ak.StructuredDataClassifier(overwrite=True, 
                                  num_classes=2,
                                  max_trials=15,
                                  )
clf.fit(
    X_train,
    y_train,
    verbose = 1,
    epochs=50
    #callbacks=[early_stopping]
)

best_model = clf.export_model()

print("Summary : ")
print(best_model.summary())

best_model.save("c2_sp2")

y_pred = clf.predict(X_test)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
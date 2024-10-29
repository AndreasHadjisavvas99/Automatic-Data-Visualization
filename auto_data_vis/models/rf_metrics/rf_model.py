from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
from process import preprocess_data
import time
import json
import os

datasets = {
    'c2_sp': '../../features1/c2_sp.csv',
    'c2_sb': '../../features1/c2_sb.csv',
    'c2_ls': '../../features1/c2_ls.csv',
    'c2_lp': '../../features1/c2_lp.csv',
    'c2_lb': '../../features1/c2_lb.csv',
    'c2_bp': '../../features1/c2_bp.csv',
    'c3_lsp': '../../features1/c3_lsp.csv',
    'c3_sbp': '../../features1/c3_sbp.csv',
    'c3_lbp': '../../features1/c3_lbp.csv',
    'c3_lsb': '../../features1/c3_lsb.csv',
    'c4_lsbp': '../../features1/c4_lsbp.csv'
}

for name, file_path in datasets.items():
    print(f"Processing dataset: {name}")

    # Load dataset
    df = pd.read_csv(file_path)
    df = df.rename(columns={'Plot_Type': 'Label'})

    # Prepare features and labels
    feature_names = df.drop('Label', axis=1).columns
    y = df["Label"].values
    X = df.drop(labels=['Label'], axis=1)

    # Preprocess data
    X = preprocess_data(X, apply_percentile_cutoff=True, apply_scaling=False, impute_missing=False, apply_one_hot_encoding=True)

    # Split data into training and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    hyperparameters = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        #'min_samples_leaf': [1, 2, 4],
        'max_features': 'sqrt',
        'criterion': 'gini',
    }

    # Initialize and train Random Forest classifier
    rf_classifier = RandomForestClassifier(**hyperparameters)

    start_time = time.time()
    rf_classifier.fit(X_train, y_train)
    training_duration = time.time() - start_time

    # Evaluate model
    val_predictions = rf_classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Evaluate the best model on the test set
    test_predictions = rf_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
 

    # Get feature importances from the best model
    importances = rf_classifier.feature_importances_
    feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"Validation accuracy for {name}: {val_accuracy:.4f}")
    print(f"Training duration: {training_duration:.2f} seconds")

    ## Print feature importance
    #print("\nFeature Importances:")
    #for feature, importance in feature_importances:
    #    print(f"{feature}: {importance:.4f}")

    #print(feature_names)



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
from process import preprocess_data
import time
import os

datasets = {
    #'c2_sp': '../../features1/c2_sp.csv',
    #'c2_sb': '../../features1/c2_sb.csv',
    #'c2_ls': '../../features1/c2_ls.csv',
    #'c2_lp': '../../features1/c2_lp.csv',
    #'c2_lb': '../../features1/c2_lb.csv',
    #'c2_bp': '../../features1/c2_bp.csv',
    #'c3_lsp': '../../features1/c3_lsp.csv',
    #'c3_sbp': '../../features1/c3_sbp.csv',
    #'c3_lbp': '../../features1/c3_lbp.csv',
    #'c3_lsb': '../../features1/c3_lsb.csv',
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
    X = preprocess_data(X, apply_percentile_cutoff=True, apply_scaling=True, impute_missing=True, apply_one_hot_encoding=True)

    # Split data into training and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    hyperparameters = {
        'C': 1, 
        'solver': 'newton-cg',
        'max_iter': 2000
    }

    lr_classifier = LogisticRegression(**hyperparameters)
    
    start_time = time.time()
    lr_classifier.fit(X_train, y_train)
    training_duration = time.time() - start_time

    # Evaluate model
    val_predictions = lr_classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Evaluate the best model on the test set
    test_predictions = lr_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Evaluate the best model on the test set
    test_predictions = lr_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Validation accuracy for {name}: {val_accuracy:.4f}")
    print(f"Training duration: {training_duration:.2f} seconds")
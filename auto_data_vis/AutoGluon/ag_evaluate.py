from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ag_process import *
import pandas as pd

# Define your models and datasets
models = {
    #'c2_bp': 'AutogluonModels/c2_bp',
    #'c2_lb': 'AutogluonModels/c2_lb',
    #'c2_lp': 'AutogluonModels/c2_lp',
    #'c2_ls': 'AutogluonModels/c2_ls',
    #'c2_sb': 'AutogluonModels/c2_sb',
    #'c2_sp': 'AutogluonModels/c2_sp',
    #'c3_lbp': 'AutogluonModels/c3_lbp',
    #'c3_lsb': 'AutogluonModels/c3_lsb',
    #'c3_lsp': 'AutogluonModels/c3_lsp',
    #'c3_sbp': 'AutogluonModels/c3_sbp',
    'c4_lsbp': 'AutogluonModels/c4_lsbp'
    #'c4_lsbp2': 'AutogluonModels/ag-20241005_180500'
}

datasets = {
    #'c2_bp': pd.read_csv('../features1/c2_bp.csv'),
    #'c2_lb': pd.read_csv('../features1/c2_lb.csv'),
    #'c2_lp': pd.read_csv('../features1/c2_lp.csv'),
    #'c2_ls': pd.read_csv('../features1/c2_ls.csv'),
    #'c2_sb': pd.read_csv('../features1/c2_sb.csv'),
    #'c2_sp': pd.read_csv('../features1/c2_sp.csv'),
    #'c3_lbp': pd.read_csv('../features1/c3_lbp.csv'),
    #'c3_lsb': pd.read_csv('../features1/c3_lsb.csv'),
    #'c3_lsp': pd.read_csv('../features1/c3_lsp.csv'),
    #'c3_sbp': pd.read_csv('../features1/c3_sbp.csv'),
    'c4_lsbp': pd.read_csv('../features1/c4_lsbp.csv'),
    #'c4_lsbp2': pd.read_csv('../features1/c4_lsbp2.csv')
}

# Iterate through each model and its corresponding dataset
for name, model_path in models.items():
    print(f"Evaluating model: {name}")
    
    # Load the model
    predictor = TabularPredictor.load(model_path)
    
    # Load and preprocess the dataset
    df = datasets[name]
    df = df.rename(columns={'Plot_Type': 'Label'})
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    X = preprocess_data(X, apply_percentile_cutoff=True, apply_scaling=False, impute_missing=False, apply_one_hot_encoding=False)
    
    # Split the data into train and test sets
    #train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=33)
    X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=0.2, random_state=100)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=100)
    
    # Make predictions on the test set
    predictions = predictor.predict(X_val)
    
    # Calculate and print the accuracy
    accuracy = accuracy_score(y_val, predictions)
    print(f"Validation Accuracy for {name}: {accuracy:.4f}\n")

    # Print the first 5 predictions
    print("First 5 Predictions:")
    print(predictions[:5].tolist())
    

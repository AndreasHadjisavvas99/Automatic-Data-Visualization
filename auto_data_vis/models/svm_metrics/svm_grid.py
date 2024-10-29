from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
from rf_process import preprocess_data
import time
import os

# Define datasets
datasets = {
    'c4_lsbp': '../features1/c4_lsbp.csv'
}

def save_selected_grid_search_results(cv_results, file_name):
    """Save selected metrics and parameters from GridSearchCV to a CSV file, sorted by validation accuracy."""
    # Extract the hyperparameters and selected metrics
    params_columns = [col for col in cv_results if col.startswith('param_')]  # Hyperparameters
    selected_columns = {
        'mean_train_accuracy': cv_results['mean_train_score'],  # Training accuracy
        'mean_test_accuracy': cv_results['mean_test_score'],    # Validation accuracy (cross-validation)
        'mean_train_loss': 1 - cv_results['mean_train_score'],  # Training loss (1 - accuracy)
        'mean_test_loss': 1 - cv_results['mean_test_score'],    # Validation loss (1 - accuracy)
    }

    # Convert the hyperparameters and metrics to a DataFrame
    params_df = pd.DataFrame(cv_results, columns=params_columns)  # Hyperparameter columns
    metrics_df = pd.DataFrame(selected_columns)                   # Selected metrics columns

    # Combine parameters and selected metrics
    combined_df = pd.concat([params_df, metrics_df], axis=1)

    # Sort the combined DataFrame by validation accuracy in descending order (best models first)
    combined_df = combined_df.sort_values(by='mean_test_accuracy', ascending=False)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Save the sorted DataFrame to a CSV file
    combined_df.to_csv(file_name, index=False)

# Process each dataset
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

    # Define parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
        'gamma': ['scale', 'auto']  # Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’
    }

    # Initialize and train SVM classifier
    svm_classifier = SVC()

    grid_search = GridSearchCV(estimator=svm_classifier,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5,  # 5-fold cross-validation
                               n_jobs=-1,  # Use all available cores
                               verbose=2,
                               return_train_score=True)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_duration = time.time() - start_time

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Evaluate the best model on the validation set
    best_svm_model = grid_search.best_estimator_
    val_predictions = best_svm_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Evaluate the best model on the test set
    test_predictions = best_svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Save metrics
    metrics = {
        "dataset": name,
        "best_params": best_params,
        "best_cross_validation_accuracy": best_score,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "training_duration_seconds": training_duration
    }    

    # Print the metrics
    print(f"Metrics for {name}:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save all trials' results (from cv_results_)
    results_filename = f'svm_metrics/{name}_grid_search_results.csv'
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    save_selected_grid_search_results(grid_search.cv_results_, results_filename)
    
print(f"Selected grid search results with parameters saved and sorted by validation accuracy to {results_filename}")

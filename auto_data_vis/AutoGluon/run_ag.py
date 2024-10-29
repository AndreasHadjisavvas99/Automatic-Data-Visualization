from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import autogluon.core as ag
from autogluon.common import space
import pandas as pd
import numpy as np
from ag_process import *

# Load your dataset
df = pd.read_csv('../features/c4_lsbp.csv')
df = df.rename(columns={'Plot_Type':'Label'})

# Separate the target variable
Y = df["Label"].values
df = df.drop(labels=['Label'], axis=1)

# Preprocess the data with your desired steps
X = preprocess_data(df, apply_percentile_cutoff=True, apply_scaling=False, impute_missing=False, apply_one_hot_encoding=False)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=100, stratify=Y)

# Convert y_train and y_test to Series and reset index
y_train = pd.Series(y_train).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)

# Create train_data and test_data by combining features with labels
train_data = pd.concat([X_train.reset_index(drop=True), y_train.rename('Label')], axis=1)
test_data = pd.concat([X_test.reset_index(drop=True), y_test.rename('Label')], axis=1)

# Specifying hyperparameters and tuning them
nn_options = {
    'num_epochs': space.Int(lower=1, upper=1000, default=200),
    'learning_rate': space.Real(lower=1e-5, upper=1e-1, default=1e-3, log=True),  # Μια σταθερή τιμή αντί για space.Real
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),   # Σταθερή τιμή αντί για space.Categorical
    'dropout_prob': space.Real(lower=0.0, upper=0.5, default=0.1),  # Σταθερή τιμή για το dropout_prob
}

gbm_options = { 
    'num_boost_round': 100, 
    'num_leaves': space.Int(lower=10, upper=30, default=15), 
}

hyperparameters = {  
                   'GBM': {},
                   'NN_TORCH': {}, 
                  } 

time_limit = 20*60  
num_trials = 100  # try at most 5 different hyperparameter configurations for each type of model
search_strategy = 'auto'  
metric = 'accuracy'

hyperparameter_tune_kwargs = { 
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}  # Refer to TabularPredictor.fit docstring for all valid values

# Train the model
predictor = TabularPredictor(label='Label', eval_metric=metric).fit(
    train_data,
    auto_stack=True,
    #tuning_data=test_data,
    time_limit=time_limit,
    #hyperparameters=hyperparameters,
    #hyperparameter_tune_kwargs=hyperparameter_tune_kwargs
    )

# Make predictions and evaluate
y_pred = predictor.predict(X_test)
print(predictor.evaluate(test_data))
print(predictor.leaderboard(test_data))

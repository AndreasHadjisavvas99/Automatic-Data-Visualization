import streamlit as st
import tensorflow as tf
import pandas as pd
from val_data import *
import matplotlib.pyplot as plt
from feature_extraction import *
from chart_function import *
from autogluon.tabular import TabularPredictor

models_ag = {
    'c2_bp': '../autogluon/AutogluonModels/c2_bp',
    'c2_lb': '../autogluon/AutogluonModels/c2_lb',
    'c2_lp': '../autogluon/AutogluonModels/c2_lp',
    'c2_ls': '../autogluon/AutogluonModels/c2_ls',
    'c2_sb': '../autogluon/AutogluonModels/c2_sb',
    'c2_sp': '../autogluon/AutogluonModels/c2_sp',
    'c3_lbp': '../autogluon/AutogluonModels/c3_lbp',
    'c3_lsb': '../autogluon/AutogluonModels/c3_lsb',
    'c3_lsp': '../autogluon/AutogluonModels/c3_lsp',
    'c3_sbp': '../autogluon/AutogluonModels/c3_sbp',
    'c4_lsbp': '../autogluon/AutogluonModels/c4_lsbp'
}
datasets = {
    'c2_bp': pd.read_csv('../features1/c2_bp.csv'),
    'c2_lb': pd.read_csv('../features1/c2_lb.csv'),
    'c2_lp': pd.read_csv('../features1/c2_lp.csv'),
    'c2_ls': pd.read_csv('../features1/c2_ls.csv'),
    'c2_sb': pd.read_csv('../features1/c2_sb.csv'),
    'c2_sp': pd.read_csv('../features1/c2_sp.csv'),
    'c3_lbp': pd.read_csv('../features1/c3_lbp.csv'),
    'c3_lsb': pd.read_csv('../features1/c3_lsb.csv'),
    'c3_lsp': pd.read_csv('../features1/c3_lsp.csv'),
    'c3_sbp': pd.read_csv('../features1/c3_sbp.csv'),
    'c4_lsbp': pd.read_csv('../features1/c4_lsbp.csv')
}

mappings = {
    'c2_bp': {0: 'bar', 1: 'pie'},
    'c2_lb': {0: 'bar', 1: 'line'},
    'c2_lp': {0: 'line', 1: 'pie'},
    'c2_ls': {0: 'line', 1: 'scatter'},
    'c2_sb': {0: 'bar', 1: 'scatter'},
    'c2_sp': {0: 'pie', 1: 'scatter'},
    'c3_lbp': {0: 'bar', 1: 'line', 2: 'pie'},
    'c3_lsb': {0: 'bar', 1: 'line', 2: 'scatter'},
    'c3_lsp': {0: 'line', 1: 'pie', 2: 'scatter'},
    'c3_sbp': {0: 'bar', 1: 'pie', 2: 'scatter'},
    'c4_lsbp': {0: 'bar', 1: 'line', 2: 'pie', 3: 'scatter'}
}

reverse_mapping = {
    frozenset(['bar', 'pie']): 'c2_bp',
    frozenset(['bar', 'line']): 'c2_lb',
    frozenset(['line', 'pie']): 'c2_lp',
    frozenset(['line', 'scatter']): 'c2_ls',
    frozenset(['bar', 'scatter']): 'c2_sb',
    frozenset(['pie', 'scatter']): 'c2_sp',
    frozenset(['bar', 'line', 'pie']): 'c3_lbp',
    frozenset(['bar', 'line', 'scatter']): 'c3_lsb',
    frozenset(['line', 'pie', 'scatter']): 'c3_lsp',
    frozenset(['bar', 'pie', 'scatter']): 'c3_sbp',
    frozenset(['bar', 'line', 'pie', 'scatter']): 'c4_lsbp'
}

st.set_page_config(layout="wide")  # Enable wide layout for full-screen mode
# Tile
st.title("My Data Visualization Recommendation App")

file = st.file_uploader("Upload data", type=["json", "csv"])
if file is not None:
    #file = json.load(file)
    if file.name.endswith('.json'):
        file = json.load(file)

        # Extract features for JSON data
        x_values, y_values, values, column_names = calc_features_json(file)
        df_values = pd.DataFrame([values], columns=column_names)
    
    # Handle CSV files
    elif file.name.endswith('.csv'):
        df_values = pd.read_csv(file)  # Directly load the CSV file into a DataFrame
        x_values,y_values, values, column_names = calc_features_csv(df_values)
        df_values = pd.DataFrame([values], columns=column_names)
    



    # User selects preferred visualizations
    preferred_viz = st.multiselect(
        "Select your preferred visualizations",
        ['bar', 'line', 'scatter', 'pie'],
        default=[]
    )
    if len(preferred_viz) < 2:
        st.warning("Please choose at least 2 visualizations.")  # Warning message
    else:
        # Determine the model based on selected visualizations
        selected_model = reverse_mapping.get(frozenset(preferred_viz))
        # Determine the model based on selected visualizations
        selected_model = reverse_mapping.get(frozenset(preferred_viz))
        
        # load predictor model
        model_path = models_ag[selected_model]
        model = TabularPredictor.load(model_path)
        df = datasets[selected_model]
        mapping = mappings[selected_model]

        #x_values,y_values,values,column_names = calc_features(file)
        #df_values = pd.DataFrame([values], columns=column_names)
        #st.write(column_names)  

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("Input:")
            st.write(pd.DataFrame({'x_values': x_values, 'y_values': y_values}))

            # Display the features with their corresponding column names
            st.write("Features:")
            st.write(df_values)
            a = preprocess_for_ag(df_values, apply_percentile_cutoff=True, apply_scaling=False, impute_missing=False, apply_one_hot_encoding=False)
            st.write("Processed Features:",a)
            prediction = model.predict(a)
            prediction_proba = model.predict_proba(a)

            plot_type_index = np.argmax(prediction_proba, axis=1)  # Get index of the highest probability
            plot_type = mapping[plot_type_index[0]] 

            st.write("Predicted Probabilities:")
            # Convert the probabilities DataFrame to a more user-friendly format
            df_probabilities = pd.DataFrame(prediction_proba)

            # Optionally, you can label the columns with your mapping
            df_probabilities.columns = [mapping[i] for i in range(len(mapping))]
            df_probabilities = df_probabilities * 100
            df_probabilities = df_probabilities.applymap(lambda x: f"{x:.2f}%")

            # Transpose the DataFrame for vertical display
            df_probabilities_transposed = df_probabilities.transpose()
            df_probabilities_transposed.columns = ['Probability']
            st.table(df_probabilities_transposed)

        with col2:
            st.write("Recommended Visualization:")
            plot_data(x_values, y_values, plot_type)
        



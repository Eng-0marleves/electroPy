import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *

def detect_task_type(column):
    if column.nunique() <= 2:
        return 'classification'
    else:
        return 'regression'

st.title('AutoML with PyCaret')

# Load Dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Let user decide columns to drop
    cols_to_drop = st.multiselect('Select columns to drop:', data.columns)
    data = data.drop(cols_to_drop, axis=1)

    # User selects target column
    target = st.selectbox('Select the target variable:', data.columns)

    task = detect_task_type(data[target])
    st.write(f"Detected task: {task}")

    # Null values handling
    categorical_cols = data.select_dtypes(['object']).columns
    continuous_cols = data.select_dtypes(['int64', 'float64']).columns

    if len(categorical_cols) > 0:
        cat_missing = st.selectbox('How would you like to handle missing values for categorical columns?', ['Most Frequent', 'Add Missing Class'])

    if len(continuous_cols) > 0:
        cont_missing = st.selectbox('How would you like to handle missing values for continuous columns?', ['Mean', 'Median', 'Mode'])
       

    if st.button('Start Training'):
        if task == 'classification':
            setup(data, target=target)
            best_model = compare_models()
            st.write(f"Best Model: {best_model}")

        elif task == 'regression':
            setup(data, target=target, session_id=123)
            best_model = compare_models()
            st.write(f"Best Model: {best_model}")


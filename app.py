import streamlit as st
import pandas as pd
import joblib

# Load models
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
active_features = joblib.load('features.pkl')

st.title("✈️ AeroGuard Predictor: Test Bench")

# 1. Upload Test CSV
uploaded_file = st.file_uploader("Upload your test CSV file", type=['csv','txt'])

if uploaded_file:
    # 1. Read the file without headers
    test_df = pd.read_csv(uploaded_file, sep='\s+', header=None)
    
    # 2. Check for length mismatch
    if len(test_df.columns) == len(active_features):
        test_df.columns = active_features
    elif len(test_df.columns) == 26:
        test_df.columns = 26
    else:
        st.warning(f"File has {len(test_df.columns)} columns, but model expects {len(active_features)}.")
       

    # 3. Select a row index
    row_idx = st.number_input("Select row index to test", min_value=0, max_value=len(test_df)-1, step=1)
    
    selected_row = test_df.iloc[[row_idx]]
    
    if st.button("Run Prediction"):
        # Extract features
        X = selected_row[active_features]
        
        # Predict
        scaled_X = scaler.transform(X)
        prediction = model.predict(scaled_X)[0]
        
        # Display Results using SAFE LOGIC
        col1, col2 = st.columns(2)
        col1.metric("Model Prediction", f"{int(prediction)} cycles")
        
        # Only try to show Ground Truth if 'RUL' somehow exists in the dataframe
        if 'RUL' in selected_row.columns:
            y_true = selected_row['RUL'].values[0]
            col2.metric("Ground Truth (Actual)", f"{int(y_true)} cycles")
            error = abs(prediction - y_true)
            st.write(f"**Absolute Error for this row:** {error:.2f}")
        else:
            col2.metric("Ground Truth (Actual)", "N/A (Not in file)")
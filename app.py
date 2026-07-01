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
    test_df = pd.read_csv(uploaded_file, sep='\s+')
    
    # 2. Select a row index
    row_idx = st.number_input("Select row index to test", min_value=0, max_value=len(test_df)-1, step=1)
    
    selected_row = test_df.iloc[[row_idx]]
    
    if st.button("Run Prediction"):
        # Extract features and ground truth
        X = selected_row[active_features]
        y_true = selected_row['RUL'].values[0] # Assuming your file has a 'RUL' column
        
        # Predict
        scaled_X = scaler.transform(X)
        prediction = model.predict(scaled_X)[0]
        
        # Display Results
        col1, col2 = st.columns(2)
        col1.metric("Model Prediction", f"{int(prediction)} cycles")
        col2.metric("Ground Truth (Actual)", f"{int(y_true)} cycles")
        
        # Show Error
        error = abs(prediction - y_true)
        st.write(f"**Absolute Error for this row:** {error:.2f}")
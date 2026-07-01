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
    # 1. Read the raw file without headers
    test_df = pd.read_csv(uploaded_file, sep='\s+', header=None)
    
    # 2. Define the standard 26 NASA CMAPSS columns
    # IMPORTANT: Ensure 's_1' or 'sensor_1' matches EXACTLY how you named them in your training notebook!
    raw_columns = [
        'engine_id','cycle','setting_1','setting_2','setting_3',
        'sensor_1','sensor_2','sensor_3','sensor_4','sensor_5',
        'sensor_6','sensor_7','sensor_8','sensor_9','sensor_10',
        'sensor_11','sensor_12','sensor_13','sensor_14','sensor_15',
        'sensor_16','sensor_17','sensor_18','sensor_19','sensor_20','sensor_21'

    ]
    
    # Check if the uploaded file is a raw 26-column NASA file
    if len(test_df.columns) == 26:
        test_df.columns = raw_columns
    elif len(test_df.columns) == len(active_features):
        test_df.columns = active_features
    else:
        st.error(f"Data shape mismatch! Uploaded file has {len(test_df.columns)} columns.")
        st.stop() # Stops the app from crashing further down

    # 3. Select a row index
    row_idx = st.number_input("Select row index to test", min_value=0, max_value=len(test_df)-1, step=1)
    
    selected_row = test_df.iloc[[row_idx]]
    
    if st.button("Run Prediction"):
        # 4. Extract ONLY the features the model needs
        X = selected_row[active_features]
        
        # Predict
        scaled_X = scaler.transform(X)
        prediction = model.predict(scaled_X)[0]
        
        # Display Results
        col1, col2 = st.columns(2)
        col1.metric("Model Prediction", f"{int(prediction)} cycles")
        
        # Safe Logic for Ground Truth
        if 'RUL' in selected_row.columns:
            y_true = selected_row['RUL'].values[0]
            col2.metric("Ground Truth (Actual)", f"{int(y_true)} cycles")
            error = abs(prediction - y_true)
            st.write(f"**Absolute Error for this row:** {error:.2f}")
        else:
            col2.metric("Ground Truth (Actual)", "N/A (Raw Data)")
            
        # Maintenance Alert Logic
        if prediction < 50:
            st.error("⚠️ Warning: Engine approaching critical status! Schedule maintenance.")
        else:
            st.success("✅ Engine status: Stable.")
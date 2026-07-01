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
        # 4. Extracting ONLY the features the model needs
        X = selected_row[active_features]
        
        # Predict current point-in-time
        scaled_X = scaler.transform(X)
        prediction = model.predict(scaled_X)[0]
        
       ## trend smoothing for visualisation
        current_unit = selected_row['engine_id'].values[0]
        current_cycle = selected_row['cycle'].values[0]
        
        # Filter all rows for this engine up to the selected cycle
        engine_history = test_df[
            (test_df['engine_id'] == current_unit) & 
            (test_df['cycle'] <= current_cycle)
        ].sort_values('cycle')
        
        # Generate predictions for the engine's entire timeline up to this point
        hist_X = engine_history[active_features]
        hist_scaled = scaler.transform(hist_X)
        engine_history['Raw Prediction'] = model.predict(hist_scaled)
        
        # Apply a rolling window average (e.g., 5 cycles) to smooth sensor noise
        engine_history['Smoothed Trend'] = engine_history['Raw Prediction'].rolling(window=5, min_periods=1).mean()
        smoothed_prediction = engine_history['Smoothed Trend'].iloc[-1]
        # ----------------------------------

        # Display Results
        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Prediction", f"{int(prediction)} cycles")
        col2.metric("Smoothed Prediction", f"{int(smoothed_prediction)} cycles")
        
        # Safe Logic for Ground Truth
        if 'RUL' in selected_row.columns:
            y_true = selected_row['RUL'].values[0]
            col3.metric("Ground Truth (Actual)", f"{int(y_true)} cycles")
            error = abs(smoothed_prediction - y_true)
            st.write(f"**Absolute Error (Smoothed):** {error:.2f}")
        else:
            col3.metric("Ground Truth (Actual)", "N/A (Raw Data)")
            
        # --- NEW: LINE CHART DISPLAY ---
        st.write(f"### 📈 RUL Timeline for Engine Unit {int(current_unit)}")
        # Change 'time_cycles' to 'cycle' here:
        chart_data = engine_history[['cycle', 'Raw Prediction', 'Smoothed Trend']].set_index('cycle')
        st.line_chart(chart_data)
        # -------------------------------
        
        # Maintenance Alert Logic (Using the smoothed prediction now)
        if smoothed_prediction < 50:
            st.error("⚠️ Warning: Engine approaching critical status! Schedule maintenance.")
        else:
            st.success("✅ Engine status: Stable.")
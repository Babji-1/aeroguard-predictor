import streamlit as st
import pandas as pd
import joblib

### Loading the exact model and pipelines
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
active_features = joblib.load('features.pkl')

st.title("✈️ AeroGuard Predictor")
st.subheader("Predictive Engine RUL Estimation")
st.write("Enter the current sensor telemetry to estimate the Remaining Useful Life (RUL).")

## Creating input fields for the active sensors
input_data = {}
cols = st.columns(3) # Organizing inputs into 3 columns for better UI

for i, feature in enumerate(active_features):
    with cols[i % 3]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# 3. Predict button
if st.button("Predict Remaining Useful Life"):
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Scale it using the saved scaler
    scaled_input = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    
    # Display result
    st.metric("Estimated Cycles Remaining", f"{int(prediction)} cycles")
    
    if prediction < 50:
        st.error("⚠️ Warning: Engine approaching critical status! Schedule maintenance immediately.")
    else:
        st.success("✅ Engine status: Stable.")
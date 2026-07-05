# ✈️ AeroGuard Predictor

## Project Description

AeroGuard Predictor is a machine learning application developed to estimate the **Remaining Useful Life (RUL)** of aircraft engines using historical engine sensor data. The project aims to support predictive maintenance by forecasting how many flight cycles remain before an engine reaches failure, helping maintenance teams make proactive maintenance decisions.

---

# Dataset

**Dataset Name:** NASA CMAPSS Turbofan Engine Remaining Useful Life Dataset (FD001)

**Source:** https://www.kaggle.com/datasets/fareselgohary003/nasa-cmapss-turbofan-engine-rul-dataset

**Dataset Size:**

| File | Rows | Columns |
|------|-----:|--------:|
| train_FD001 | 20,631 | 26 |
| test_FD001 | 13,096 | 26 |
| RUL_FD001 | 100 | 1 |

**Key Features**

- Engine ID
- Flight Cycle
- 3 Operational Settings
- 21 Sensor Measurements
- Remaining Useful Life (generated for training)

---

# Tech Stack

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

# How to Run

1. Clone the repository.

```bash
git clone https://github.com/Babji-1/aeroguard-predictor.git
```

2. Navigate into the project folder.

```bash
cd aeroguard-predictor
```

3. Install the required packages.

```bash
pip install -r requirements.txt
```

4. Launch the Streamlit application.

```bash
streamlit run app.py
```

5. Upload a CSV file containing engine sensor data to generate Remaining Useful Life predictions.

---

# Key Findings

- XGBoost achieved the best overall performance among the evaluated models.
- The final deployed model achieved a **Mean Absolute Error (MAE) of 23.45 flight cycles**.
- Proper preprocessing, including engine-level train-validation splitting and scaling after splitting, helped prevent data leakage.
- The deployed Streamlit application allows users to upload engine sensor data and receive Remaining Useful Life predictions with a visual prediction timeline.
- Predictive maintenance can help reduce unexpected engine failures and improve maintenance planning.

---

## 🔗 Project Resources

| Resource | Link |
|----------|------|
| 🌐 Live Application | [AeroGuard Predictor (Streamlit)](https://aeroguard-predictor.streamlit.app/) |
| 📊 Tableau Dashboard | [Interactive Dashboard](https://public.tableau.com/app/profile/kennedy.kiiru/viz/Aeroguard/Dashboard1) |
| 🎤 Project Presentation | [Canva Presentation](https://canva.link/e9ryb48krkqfrl8) |
| 💻 Source Code | [GitHub Repository](https://github.com/Babji-1/aeroguard-predictor) |

# A Project by:

**Kennedy Kiiru**

GitHub: https://github.com/Babji-1

---

# License

This project is licensed under the **MIT License**.



# ✈️ AeroGuard Predictor
## Predicting Aircraft Engine Remaining Useful Life (RUL) using Machine Learning



---

## 📖 Table of Contents

- Project Overview
- Business Problem
- Project Objectives
- Technical Stack
- Dataset
- Project Workflow
- Exploratory Data Analysis
- Feature Engineering
- Data Leakage Prevention
- Machine Learning Models
- Model Performance
- Deployment
- Repository Structure
- Installation
- Usage
- Future Improvements
- Real-World Impact
- Acknowledgements
- Author

---

# 📌 Project Overview

**AeroGuard Predictor** is an end-to-end Machine Learning application designed to estimate the **Remaining Useful Life (RUL)** of commercial aircraft engines using historical engine sensor measurements.

The project leverages supervised machine learning techniques to analyze operational sensor data collected throughout an engine's lifecycle and estimate how many flight cycles remain before the engine reaches failure. Instead of relying solely on fixed maintenance schedules or waiting for mechanical failures, AeroGuard demonstrates how predictive analytics can support proactive maintenance planning.

Using the **NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**, multiple regression models were trained and evaluated to predict engine degradation. The best-performing model was then deployed as an interactive **Streamlit web application**, allowing users to upload engine sensor data and receive instant Remaining Useful Life predictions along with a visual prediction timeline.

This project demonstrates the complete lifecycle of a production-oriented Machine Learning solution, including:

- Business problem understanding
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data leakage prevention
- Model training and comparison
- Performance evaluation
- Model deployment using Streamlit

Rather than being just another notebook, AeroGuard represents a practical predictive maintenance solution capable of assisting maintenance engineers in making informed, data-driven decisions.

---

# 💼 Business Problem

Aircraft engine maintenance has traditionally followed two major strategies:

### Reactive Maintenance

Maintenance occurs only after a component has failed.

While inexpensive in the short term, reactive maintenance can lead to:

- Unexpected aircraft downtime
- Increased repair costs
- Flight delays
- Safety concerns
- Reduced fleet availability

---

### Preventive Maintenance

Maintenance is performed after a fixed number of operating hours or flight cycles regardless of the engine's actual health.

Although safer than reactive maintenance, this approach often replaces components that still possess significant operational life, leading to:

- Unnecessary maintenance costs
- Increased spare part consumption
- Reduced asset utilization
- Higher operational expenditure

---

## The Predictive Maintenance Approach

Predictive Maintenance combines sensor data with Machine Learning algorithms to estimate an engine's Remaining Useful Life before failure occurs.

Instead of asking:

> "Has this engine failed?"

the model asks:

> **"How many flight cycles does this engine still have before failure?"**

By accurately estimating Remaining Useful Life (RUL), maintenance teams can:

- Schedule maintenance proactively
- Reduce unexpected failures
- Improve aircraft availability
- Minimize maintenance costs
- Improve operational planning
- Enhance aviation safety

AeroGuard Predictor was developed to demonstrate how Machine Learning can transform raw engine sensor data into actionable maintenance intelligence.

---

# 🎯 Project Objectives

The primary objective of this project was to develop a predictive maintenance system capable of accurately estimating aircraft engine Remaining Useful Life using Machine Learning.

Specific objectives included:

- Build an end-to-end Machine Learning pipeline for predictive maintenance.

- Generate Remaining Useful Life (RUL) labels from engine lifecycle data.

- Perform exploratory data analysis to understand engine degradation patterns.

- Identify and retain the most informative engine sensor measurements.

- Compare multiple regression algorithms for RUL prediction.

- Evaluate model performance using industry-standard regression metrics.

- Prevent data leakage through proper preprocessing and dataset separation.

- Deploy the best-performing model as an interactive Streamlit web application.

- Demonstrate how predictive maintenance can improve operational efficiency within the aviation industry.

---

# 🛠 Technical Stack

### Programming Language

- Python 3.13

---

### Data Processing

- Pandas
- NumPy

---

### Data Visualization

- Matplotlib
- Seaborn

---

### Machine Learning

- Scikit-Learn
- XGBoost

---

### Model Serialization

- Joblib

---

### Web Deployment

- Streamlit

---

### Version Control

- Git
- GitHub

---

# 🔗 Project Links

### 🌐 Live Streamlit Application

https://aeroguard-predictor.streamlit.app/

---

### 🎤 Project Presentation (Canva)

https://canva.link/e9ryb48krkqfrl8

---

### 📊 Tableau Dashboard

*Coming Soon*

---

### 📁 GitHub Repository

https://github.com/Babji-1/aeroguard-predictor
# 📂 Dataset

This project uses the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** Turbofan Engine Degradation Dataset, one of the most widely used benchmark datasets for predictive maintenance and Remaining Useful Life (RUL) estimation.

The dataset contains simulated run-to-failure records for multiple aircraft engines operating under varying environmental conditions and fault scenarios. Each engine begins in a healthy state and gradually degrades until failure, allowing machine learning models to learn degradation patterns from historical sensor measurements.

**Dataset Source:**

🔗 https://www.kaggle.com/datasets/fareselgohary003/nasa-cmapss-turbofan-engine-rul-dataset

---

## Dataset Files

The project utilizes the following files from the FD001 subset:

| File | Description |
|------|-------------|
| `train_FD001.txt` | Historical run-to-failure engine data used for training the models. |
| `test_FD001.txt` | Partial engine operating histories representing active engines that have not yet failed. |
| `RUL_FD001.txt` | Ground truth Remaining Useful Life values for each engine in the test dataset. |

---

## Dataset Characteristics

- Multiple aircraft engines
- Multiple flight cycles per engine
- 21 onboard sensor measurements
- 3 operational setting variables
- Progressive engine degradation
- Simulated engine failures
- Ground truth Remaining Useful Life labels

Each row in the dataset represents the operational condition of an engine during a specific flight cycle.

---

## Target Variable

The prediction target for this project is:

> **Remaining Useful Life (RUL)**

Remaining Useful Life represents the estimated number of flight cycles remaining before an engine reaches failure.

Example:

| Current Flight Cycle | Failure Cycle | Remaining Useful Life |
|---------------------:|--------------:|----------------------:|
| 120 | 200 | 80 |
| 150 | 200 | 50 |
| 180 | 200 | 20 |

As engines age, the Remaining Useful Life decreases until it eventually reaches zero.

---

# 🔄 Project Workflow

The AeroGuard Predictor follows an end-to-end Machine Learning workflow, beginning with raw sensor measurements and ending with an interactive prediction application.

```text
                    NASA CMAPSS Dataset
                            │
                            ▼
                 Data Cleaning & Inspection
                            │
                            ▼
              Remaining Useful Life (RUL) Generation- Feature Engineering
                            │
                            ▼
             Exploratory Data Analysis (EDA)
                            │
                            ▼
              Sensor Selection & Feature Engineering
                            │
                            ▼
         Train / Validation Split (Grouped by Engine)
                            │
                            ▼
             Data Scaling (Training Data Only)
                            │
                            ▼
             Machine Learning Model Training
                            │
                            ▼
          Model Comparison & Performance Evaluation
                            │
                            ▼
             Final Testing on Unseen NASA Test Data
                            │
                            ▼
                 Model Serialization (Joblib)
                            │
                            ▼
                Streamlit Web Application
```

This workflow follows standard Machine Learning practices by ensuring preprocessing is performed correctly, preventing data leakage and preserving the integrity of model evaluation.

---

# 📊 Exploratory Data Analysis

Before training any Machine Learning models, exploratory data analysis was conducted to better understand the dataset, identify potential issues, and uncover relationships between engine sensor measurements and Remaining Useful Life.

The analysis focused on understanding engine degradation behaviour rather than immediately training predictive models.

The following analyses were performed:

## Engine Degradation Trends

Engine sensor measurements were visualized across flight cycles to observe how engine health changes over time.

This helped reveal gradual degradation patterns that are indicative of approaching engine failure.

<img width="678" height="383" alt="image" src="https://github.com/user-attachments/assets/70a37f14-b816-48eb-85ce-1a94ca78280f" />


---

## Remaining Useful Life Distribution

The generated Remaining Useful Life labels were analyzed to understand the distribution of engine health throughout the training dataset.

This ensured that the model was exposed to engines at different stages of degradation.

<img width="167" height="387" alt="image" src="https://github.com/user-attachments/assets/23bd3977-bb0c-49ed-8dcc-e140ef0e0260" />


---

##   Feature importance

A Feature importance Br graph was generated to identify relationships among sensor variables.


<img width="1001" height="604" alt="image" src="https://github.com/user-attachments/assets/03ae2100-e657-48c3-a3e3-de55cd08fe52" />


---

## Missing Value Inspection

The dataset was inspected for missing values prior to model development.

No missing data were observed within the selected dataset subset.

---

# ⚙️ Feature Engineering

Feature engineering plays a critical role in predictive maintenance by transforming raw operational data into meaningful model inputs.

Several preprocessing and feature engineering steps were performed to improve model performance and reduce noise.

---

## Remaining Useful Life Generation

The original NASA dataset does not directly provide Remaining Useful Life values for every training observation.

Instead, RUL labels were generated by calculating:

```text
Failure Cycle − Current Cycle
```

This transformation converted the dataset into a supervised regression problem.

---

## Active Sensor Selection

Not every sensor contributes meaningful information about engine degradation.

Sensors exhibiting little or no variation across engine lifecycles were removed from the feature set.

Only informative sensor measurements were retained for model training.

This reduced noise and improved computational efficiency.

---

## Feature Scaling

Sensor measurements operate on different numerical scales.

To ensure fair model learning, numerical features were normalized using **Min-Max Scaling**, allowing all selected sensors to contribute proportionally during training.

Importantly, the scaler was fitted **only on the training data** before being applied to validation and testing datasets.

This prevented information leakage from unseen data.

---

## Engine-Level Dataset Splitting

Rather than randomly splitting individual observations, the dataset was divided using unique **engine IDs**.

This ensured that all flight cycles belonging to a particular engine remained within the same dataset partition.

By separating engines rather than individual records, the validation data more accurately represented completely unseen engines encountered in real-world deployment.

This approach significantly reduced data leakage and produced more reliable performance estimates.

---

# 🛡️ Data Leakage Prevention

Preventing data leakage was one of the primary considerations during model development.

Data leakage occurs when information from validation or test data unintentionally influences the training process, resulting in overly optimistic performance estimates that cannot be replicated in production.

To ensure fair model evaluation, the following safeguards were implemented:

✅ Engine-level train/validation splitting

✅ Data scaling performed **after** dataset splitting

✅ Validation data remained unseen during training

✅ Final NASA test dataset reserved exclusively for final model evaluation

These practices ensure that reported performance metrics more accurately reflect how the model would perform on entirely new aircraft engines in real operational environments.
# 🤖 Machine Learning Models

AeroGuard Predictor was developed as a **supervised regression problem**, where the objective is to predict a continuous numerical value: the **Remaining Useful Life (RUL)** of an aircraft engine.

Rather than relying on a single algorithm, three different regression models were trained and evaluated to determine which provided the most accurate predictions.

---

## 1️⃣ Linear Regression

Linear Regression was selected as the baseline model due to its simplicity and interpretability.

It assumes a linear relationship between the selected engine sensor measurements and the Remaining Useful Life.

Although aircraft engine degradation is highly nonlinear, Linear Regression provides an excellent benchmark against which more advanced algorithms can be compared.

### Advantages

- Fast training
- Easy to interpret
- Computationally efficient
- Establishes a baseline for comparison

---

## 2️⃣ Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines the predictions of multiple decision trees.

Instead of relying on a single model, Random Forest aggregates predictions from many trees, allowing it to capture nonlinear relationships between engine sensor measurements and Remaining Useful Life.

### Advantages

- Captures complex nonlinear patterns
- Robust against overfitting
- Handles noisy sensor data well
- Requires minimal feature engineering

---

## 3️⃣ XGBoost Regressor

Extreme Gradient Boosting (XGBoost) is a powerful ensemble algorithm that builds trees sequentially, with each new tree learning from the errors made by previous trees.

Unlike Random Forest, which trains trees independently, XGBoost continuously improves its predictions by correcting previous mistakes.

This makes it particularly effective for structured tabular datasets such as NASA's CMAPSS engine data.

### Advantages

- Excellent predictive accuracy
- Handles nonlinear relationships effectively
- Regularization reduces overfitting
- Industry-standard algorithm for structured data
- Highly optimized and computationally efficient

---

## Why Multiple Models?

Training multiple algorithms allows objective comparison rather than assuming one model is automatically superior.

Each model offers different strengths:

| Model | Primary Purpose |
|--------|-----------------|
| Linear Regression | Baseline performance |
| Random Forest | Nonlinear ensemble learning |
| XGBoost | High-performance gradient boosting |

The model producing the lowest prediction error on unseen data was selected as the final deployment model.

---

# 📈 Model Performance

Model performance was evaluated using two widely accepted regression metrics:

### Mean Absolute Error (MAE)

MAE measures the average prediction error in **flight cycles**.

For example, an MAE of **23 cycles** means the model's predictions are, on average, approximately 23 flight cycles away from the actual engine Remaining Useful Life.

---

### Root Mean Squared Error (RMSE)

RMSE also measures prediction error in flight cycles but penalizes large prediction mistakes more heavily than MAE.

This metric is particularly valuable in aviation because large prediction errors can have significant operational and safety implications.

---

## Final Model Performance

| Model | MAE ↓ | RMSE ↓ |
|--------|-------:|--------:|
| Linear Regression | **25.65 cycles** | **32.17 cycles** |
| Random Forest | **24.23 cycles** | **33.11 cycles** |
| 🏆 XGBoost | **23.45 cycles** | **31.91 cycles** |

---

## Best Performing Model

Among the evaluated algorithms, **XGBoost achieved the lowest Mean Absolute Error and Root Mean Squared Error**, making it the most accurate predictor of Remaining Useful Life.

Although the performance improvements over Random Forest were modest, XGBoost consistently demonstrated superior generalization on unseen engine data.

For this reason, XGBoost was selected as the final production model and deployed within the Streamlit application.

---

# 🚀 Deployment

To demonstrate practical applicability beyond a Jupyter Notebook, the trained XGBoost model was deployed as an interactive web application using **Streamlit**.

The application enables users to upload aircraft engine sensor data and receive immediate Remaining Useful Life predictions without writing any Python code.

---

## Application Features

✅ Upload engine sensor data in CSV format

✅ Instant Remaining Useful Life prediction

✅ Interactive prediction timeline visualization

✅ Smoothed prediction trend

✅ Engine health status indicator

The deployment transforms the Machine Learning model into an accessible decision-support tool suitable for maintenance personnel and technical stakeholders.

---

## Live Demo

🌐 **Streamlit Application**

https://aeroguard-predictor.streamlit.app/

---

## Deployment Preview

> Replace the image path below with your screenshot location.

```markdown
![AeroGuard Streamlit App](images/streamlit_dashboard.png)
```

The deployed interface provides:

- Raw Remaining Useful Life prediction
- Smoothed prediction trend
- Interactive prediction graph
- Engine operational status
- User-friendly upload interface

---

# 📂 Repository Structure

```text
aeroguard-predictor/
│
├── notebooks/
│   └── AeroGuard_Predictor.ipynb
│
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
│
├── app.py
├── model.pkl
├── scaler.pkl
├── features.pkl
│
├── requirements.txt
├── README.md


```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/Babji-1/aeroguard-predictor.git
```

Navigate into the project directory:

```bash
cd aeroguard-predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will launch locally in your default web browser.

---

# ▶️ Usage

Using AeroGuard Predictor requires only four simple steps:

### Step 1

Launch the Streamlit application.

---

### Step 2

Upload a CSV file containing engine sensor measurements.

---

### Step 3

Allow the model to process the uploaded data.

---

### Step 4

Review the predicted Remaining Useful Life, engine health status, and interactive prediction timeline.

The application is designed for ease of use, enabling non-technical users to interact with the predictive model through a clean graphical interface.
# 🔮 Future Improvements

Although AeroGuard Predictor successfully demonstrates the application of Machine Learning to predictive aircraft maintenance, several enhancements could further improve its performance, scalability, and real-world applicability.

### Planned Improvements

- Implement Deep Learning models such as Long Short-Term Memory (LSTM) networks to better capture temporal degradation patterns.
- Perform hyperparameter optimization using Grid Search or Bayesian Optimization to further improve predictive performance.
- Integrate Explainable AI techniques such as SHAP to provide interpretable model predictions.
- Build an automated MLOps pipeline for continuous model training, validation, and deployment.
- Enable cloud deployment using platforms such as Microsoft Azure or AWS.
- Incorporate real-time IoT sensor streaming for continuous engine health monitoring.
- Expand the application to support additional NASA CMAPSS subsets and multiple engine operating conditions.
- Develop maintenance scheduling recommendations based on predicted Remaining Useful Life.

---

# 🌍 Real-World Impact

AeroGuard Predictor demonstrates how Machine Learning can support data-driven decision-making within the aviation industry.

If integrated into existing airline maintenance systems, the solution has the potential to:

- Reduce unexpected aircraft downtime.
- Improve fleet availability.
- Lower maintenance and operational costs.
- Optimize spare parts inventory.
- Improve maintenance planning efficiency.
- Support proactive maintenance scheduling.
- Enhance passenger safety through earlier fault detection.
- Increase aircraft operational reliability.

Rather than replacing maintenance engineers, AeroGuard is intended to serve as a **decision-support system**, providing predictive insights that complement engineering expertise and established aviation maintenance procedures.

---

# 💡 Recommendations

Based on the findings of this project, the following recommendations are proposed:

### 🚀 Deploy the Predictive Model Through Streamlit

Deploy the trained Machine Learning model as a lightweight Streamlit web application, enabling maintenance personnel to upload engine sensor data and obtain Remaining Useful Life predictions without interacting with Python code.

---

### ⚠️ Implement Automated Operational Risk Classification

Convert numerical Remaining Useful Life predictions into operational risk categories to simplify maintenance decision-making.

| Risk Level | Remaining Useful Life | Recommended Action |
|------------|----------------------:|--------------------|
| 🟢 Normal Operations | > 50 cycles | Continue routine monitoring |
| 🟡 Maintenance Planning | 31–50 cycles | Schedule maintenance inspection |
| 🔴 Critical (AOG) | ≤ 30 cycles | Immediate maintenance intervention |

---

### 🔄 Establish a Continuous Feedback Loop

Continuously improve model performance by periodically retraining the predictive model using newly collected engine operational data and confirmed maintenance outcomes.

As aircraft fleets evolve, continuous retraining helps maintain prediction accuracy across changing operational environments.

---

### 📈 Monitor Model Performance After Deployment

Once deployed, prediction accuracy should be monitored by comparing predicted Remaining Useful Life against actual maintenance outcomes.

Any significant reduction in predictive performance should trigger model review and retraining.

---

### 👨‍🔧 Keep Human Engineers in the Decision Loop

Machine Learning predictions should support—not replace—engineering judgment.

Final maintenance decisions should continue to follow aviation safety regulations, manufacturer recommendations, and certified maintenance procedures.

---

# 🙏 Acknowledgements

This project would not have been possible without the availability of high-quality open datasets and open-source Machine Learning libraries.

Special thanks to:

- **NASA Ames Research Center** for developing the CMAPSS Turbofan Engine Degradation Simulation Dataset.
- The developers and contributors of **Python**, **Pandas**, **NumPy**, **Scikit-Learn**, **XGBoost**, **Matplotlib**, **Seaborn**, and **Streamlit**.
- **Kaggle** for hosting the dataset and making it easily accessible to the Machine Learning community.
- **Eldohub** for providing the learning environment and guidance throughout the Data Science program.

---

# 👨‍💻 Author

## Kennedy Kiiru

**Aspiring Data Scientist | Machine Learning Enthusiast**

### GitHub

https://github.com/Babji-1

---

### Project Repository

https://github.com/Babji-1/aeroguard-predictor

---

### Live Streamlit Application

[Streamlit](https://aeroguard-predictor.streamlit.app/)

---

### Canva Presentation

[Canva](https://canva.link/e9ryb48krkqfrl8)

---

### Tableau Dashboard

[Tableau](https://public.tableau.com/app/profile/kennedy.kiiru/viz/Aeroguard/Dashboard1?publish=yes)

---

# ⭐ If You Found This Project Interesting...

If you enjoyed exploring AeroGuard Predictor or found the implementation useful, consider giving the repository a ⭐ on GitHub.

Feedback, suggestions, and contributions are always welcome as I continue learning and improving my skills in Data Science and Machine Learning.

---

# 📜 License

This project is currently released **without a formal license**.

Please contact the author before redistributing or using significant portions of this work.

---

## Final Thoughts

Predictive maintenance represents one of the most impactful applications of Machine Learning in modern engineering.

By combining historical sensor measurements with data-driven predictive models, organizations can move beyond reactive maintenance strategies and toward intelligent, proactive decision-making.

AeroGuard Predictor demonstrates this concept through an end-to-end Machine Learning workflow encompassing data preprocessing, feature engineering, model development, evaluation, deployment, and interactive visualization.

While this project was developed for educational purposes, the methodologies employed closely reflect practices used in real-world predictive maintenance systems and provide a strong foundation for future expansion into production-grade Machine Learning solutions.

---

<p align="center">

**✈️ AeroGuard Predictor**

*Predicting Tomorrow's Maintenance Decisions Today.*

Made with ❤️, Python, and Machine Learning.

</p>

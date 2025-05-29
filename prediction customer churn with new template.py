import os
import json
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from xgboost import XGBClassifier

st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .prediction-box {
        background-color: #d3d3d3;  /* Slightly darker gray for better theme integration */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
        color: #2c3e50;  /* Dark text color for contrast */
    }
    .prediction-box strong {
        color: #2c3e50;  /* Ensure strong tags (bold text) are also dark */
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .metrics-table th, .metrics-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .metrics-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-header">Customer Churn Prediction</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Customer Details")

    st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])


    st.markdown('<div class="section-header">Services</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])


    st.markdown('<div class="section-header">Billing Details</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

st.write("### Model Training and Evaluation")

kaggle_api_token = {
    "username": st.secrets["kaggle"]["username"],
    "key": st.secrets["kaggle"]["key"]
}

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(kaggle_api_token, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

with st.spinner("Downloading dataset..."):
    os.system("kaggle datasets download -d blastchar/telco-customer-churn --unzip -p ./data")

df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

scaler = MinMaxScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model_path = "churn_model_xgb.pkl"
metrics_path = "model_metrics.json"

if not os.path.exists(model_path):
    with st.spinner("Training model..."):
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        
        y_pred = model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred)
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        st.write("Model Evaluation Results (computed during training):")
        st.write(metrics)


model = joblib.load(model_path)

if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    st.write("#### Model Evaluation Metrics (XGBoost)")
    
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Value']
    
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")

    metrics_html = metrics_df.to_html(index=False, classes='metrics-table')
    st.markdown(metrics_html, unsafe_allow_html=True)
else:
    st.write("Model evaluation metrics are not available. Please train the model first.")


st.write("### Predict Customer Churn")
if st.button("Predict Churn"):
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]


    st.markdown(f"""
    <div class="prediction-box">
        <strong>Churn Prediction:</strong> {'Yes' if prediction == 1 else 'No'}<br>
        <strong>Probability of Churn:</strong> {probability:.2%}
    </div>
    """, unsafe_allow_html=True)

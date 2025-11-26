# app_modern_card_ui.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

# One-hot encode categorical columns
categorical_cols = ["cp", "slope", "restecg", "ca", "thal"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Features and target
feature_columns = [col for col in df_encoded.columns if col != "target"]
X = df_encoded[feature_columns]
y = df_encoded["target"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("💓 Heart Disease Predictor")
st.markdown(f"<p style='color: gray;'>Model Accuracy: {acc*100:.2f}%</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Modern card-style input
# -------------------------------
st.subheader("Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 29, 77, 54)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_val = 0 if sex=="Female" else 1
    trestbps = st.number_input("Resting BP", 94, 200, 130)

with col2:
    chol = st.number_input("Cholesterol", 126, 564, 250)
    thalach = st.number_input("Max Heart Rate", 71, 202, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang_val = 0 if exang=="No" else 1

with col3:
    oldpeak = st.number_input("ST Depression", 0.0, 6.2, 1.0, step=0.1)
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)

restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
restecg_val = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

ca = st.selectbox("Number of Major Vessels", ["0","1","2","3","4"])
ca_val = int(ca)

thal = st.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])
thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# -------------------------------
# Prepare input dataframe
# -------------------------------
data = {
    "age": age,
    "sex": sex_val,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "exang": exang_val,
    "oldpeak": oldpeak
}
input_df = pd.DataFrame(data, index=[0])

# One-hot encode categorical features
cat_features = {
    "cp": cp_val,
    "slope": slope_val,
    "restecg": restecg_val,
    "ca": ca_val,
    "thal": thal_val
}
for feature, value in cat_features.items():
    n_values = df[feature].nunique()
    for i in range(n_values):
        col_name = f"{feature}_{i}"
        input_df[col_name] = 1 if value==i else 0

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_columns]

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.markdown("---")
    if prediction[0]==1:
        st.markdown(
            "<div style='padding:20px; background-color:#ffdddd; color:#b30000; border-radius:10px; font-size:22px;'>Heart Disease: YES</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='padding:20px; background-color:#ddffdd; color:#006600; border-radius:10px; font-size:22px;'>Heart Disease: NO</div>",
            unsafe_allow_html=True
        )

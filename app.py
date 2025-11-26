# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Animated background & modern CSS
# -------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #89f7fe, #66a6ff, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }
    .stContainer, .css-1d391kg {
        background: rgba(255,255,255,0.85);
        border-radius: 20px;
        padding: 20px;
    }
    .stButton>button {
        background-color:#4CAF50;color:white;border-radius:10px;padding:10px 20px;
        font-size:16px;transition:0.3s;
    }
    .stButton>button:hover {background-color:#45a049;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Load dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

st.title("Heart Disease Prediction App")

if st.checkbox("Show Dataset"):
    st.dataframe(df)

# -------------------------------
# 2. Prepare data
# -------------------------------
# Use original features (not encoded)
features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")

# -------------------------------
# 4. User input
# -------------------------------
st.sidebar.header("Enter Patient Data")

def user_input():
    data = {
        "age": st.sidebar.slider("Age", 29, 77, 54),
        "sex": st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", (0, 1)),
        "cp": st.sidebar.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3)),
        "trestbps": st.sidebar.slider("Resting Blood Pressure", 94, 200, 130),
        "chol": st.sidebar.slider("Cholesterol", 126, 564, 250),
        "fbs": st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1)),
        "restecg": st.sidebar.selectbox("Resting ECG (0-2)", (0, 1, 2)),
        "thalach": st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150),
        "exang": st.sidebar.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", (0, 1)),
        "oldpeak": st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0),
        "slope": st.sidebar.selectbox("Slope of ST Segment (0-2)", (0, 1, 2)),
        "ca": st.sidebar.selectbox("Number of Major Vessels (0-4)", (0, 1, 2, 3, 4)),
        "thal": st.sidebar.selectbox("Thalassemia (0-3)", (0, 1, 2, 3))
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# -------------------------------
# 5. Prediction
# -------------------------------
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("Prediction")
st.write("Heart Disease" if prediction==1 else "No Heart Disease")

st.subheader("Prediction Probability")
st.write({
    "No Heart Disease": f"{prediction_proba[0]*100:.2f}%",
    "Heart Disease": f"{prediction_proba[1]*100:.2f}%"
})

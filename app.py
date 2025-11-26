# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load the dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")  # make sure heart.csv is in the same folder
    return df

df = load_data()

st.title("Heart Disease Prediction App")

# -------------------------------
# 2. Show dataset
# -------------------------------
if st.checkbox("Show Dataset"):
    st.write(df)

# -------------------------------
# 3. Prepare data
# -------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. Train the model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc*100:.2f}%")

# -------------------------------
# 5. User input for prediction
# -------------------------------
st.sidebar.header("Enter Patient Data")

def user_input():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", (0, 1))
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
    trestbps = st.sidebar.slider("Resting Blood Pressure", 94, 200, 130)
    chol = st.sidebar.slider("Cholesterol", 126, 564, 250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", (0, 1, 2))
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", (0, 1))
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment (0-2)", (0, 1, 2))
    ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox("Thalassemia (0-3)", (0, 1, 2, 3))

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input()

# -------------------------------
# 6. Prediction
# -------------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("Heart Disease" if prediction[0]==1 else "No Heart Disease")

st.subheader("Prediction Probability")
st.write(prediction_proba)

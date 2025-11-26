import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import io

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Predictor")
st.write("Enter patient details in the sidebar to see an advanced prediction with visual insights.")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("heart.csv").drop_duplicates()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=4,
        max_depth=None,
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_model()

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Patient Details")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = 1 if st.sidebar.selectbox("Sex", ["Male", "Female"]) == "Male" else 0

    cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}
    cp = cp_map[st.sidebar.selectbox("Chest Pain Type", list(cp_map.keys()))]

    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 246)

    fbs = 1 if st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl?", ["No","Yes"])=="Yes" else 0
    restecg = st.sidebar.selectbox("Resting ECG Results", [0,1,2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = 1 if st.sidebar.radio("Exercise Induced Angina?", ["No","Yes"])=="Yes" else 0
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST", [0,1,2])
    ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal_map = {"Null":0,"Fixed Defect":1,"Normal":2,"Reversible Defect":3}
    thal = thal_map[st.sidebar.selectbox("Thalassemia", list(thal_map.keys()))]

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ----------------------------
# DASHBOARD METRIC CARDS
# ----------------------------
st.subheader("Key Patient Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age", input_df['age'].values[0])
col2.metric("Max Heart Rate", input_df['thalach'].values[0])
col3.metric("Resting BP", input_df['trestbps'].values[0])
col4.metric("Cholesterol", input_df['chol'].values[0])

# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.subheader("Heart Disease Risk Prediction")

if st.button("Predict Risk"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]*100

    # Determine risk level and color
    if probability < 30:
        color = "green"
        level = "LOW"
    elif probability < 70:
        color = "yellow"
        level = "MEDIUM"
    else:
        color = "red"
        level = "HIGH"

    # Gauge visualization using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Heart Disease Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': color},
               'steps': [
                   {'range': [0, 30], 'color': "green"},
                   {'range': [30, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "red"}]}))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Risk Level:** {level}")

    # Risk explanation
    explanations = {
        "LOW": "Low risk – maintain healthy lifestyle, routine checkups recommended.",
        "MEDIUM": "Medium risk – consider medical consultation and lifestyle review.",
        "HIGH": "High risk – consult a healthcare professional promptly."
    }
    st.info(explanations[level])

    # Comparison chart vs healthy ranges
    st.subheader("Comparison with Healthy Ranges")
    metrics = {
        "Resting BP": {"value": input_df['trestbps'].values[0], "normal": 120},
        "Cholesterol": {"value": input_df['chol'].values[0], "normal": 200},
        "Max Heart Rate": {"value": input_df['thalach'].values[0], "normal": 150}
    }
    comp_df = pd.DataFrame(metrics).T
    comp_df.reset_index(inplace=True)
    comp_df.rename(columns={"index":"Metric"}, inplace=True)
    comp_df.plot(kind="bar", x="Metric", y=["value","normal"], title="Patient vs Healthy")
    st.bar_chart(comp_df.set_index("Metric")[["value","normal"]])

    # Downloadable report
    report_df = input_df.copy()
    report_df['Predicted Probability (%)'] = round(probability,1)
    report_df['Risk Level'] = level
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Professional Report",
        data=csv_buffer.getvalue(),
        file_name="heart_disease_report.csv",
        mime="text/csv"
    )

    st.caption("AI-based estimation — not a substitute for professional medical advice.")

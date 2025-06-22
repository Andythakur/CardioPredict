import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import matplotlib.pyplot as plt
import base64
import time
from PIL import Image


@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df


def preprocess_data(df):
    # Convert categorical variables if needed (already encoded here)
    # Example: chest pain type and slope as categorical if needed
    # For simplicity assume dataset is clean

    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


# Model Training

@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return model, X_test, y_test, report, conf_matrix, roc_auc

def get_recommendations(has_disease, patient_features):
    if not has_disease:
        return {
            "message": "No heart disease detected. Maintain a healthy lifestyle and regular checkups."
        }

    modern_treatment = [
        "Consult a cardiologist for detailed evaluation.",
        "Medications: Statins, Beta-blockers, ACE inhibitors as prescribed.",
        "Lifestyle changes: Quit smoking, adopt regular exercise, healthy diet."
    ]
    ayurvedic_remedies = [
        "Take Arjuna (Terminalia arjuna) bark powder 3g daily with warm water.",
        "Use Ashwagandha supplements to reduce stress.",
        "Follow a heart-healthy diet low in oil, salt, and processed food.",
        "Practice Pranayama and Yoga daily to improve cardiovascular health."
    ]

    return {
        "Modern Treatment": modern_treatment,
        "Ayurvedic Remedies": ayurvedic_remedies
    }


def plot_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.subheader("Feature Importance and SHAP Summary Plot")
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def inject_custom_css():
    custom_css = """
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
            padding: 20px;
            border-radius: 10px;
            animation: fadeIn 1s ease-in-out;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        @keyframes fadeIn {
            0% {opacity: 0;}
            100% {opacity: 1;}
        }
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #c0392b;
        }
        .report-preview img {
            border: 2px solid #ccc;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            animation: slideFadeIn 1.2s ease;
        }
        @keyframes slideFadeIn {
            0% {
                transform: translateY(20px);
                opacity: 0;
            }
            100% {
                transform: translateY(0);
                opacity: 1;
            }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def main():
    inject_custom_css()

    logo = "logo.png"
    try:
        image = Image.open(logo)
        st.image(image, width=100)
    except:
        pass

    st.title("Heart Health Risk & Recommendation System ❤️‍🩹")
    st.write("""
    This system predicts the risk of heart disease based on patient vitals and gives treatment recommendations combining modern and Ayurvedic approaches.
    """)

    st.subheader("📄 Upload Patient Report Image (Optional)")
    uploaded_file = st.file_uploader("Upload Report Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.markdown('<div class="report-preview">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Uploaded Report", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("Report uploaded. Please manually input the detected vitals below.")

    df = load_data()
    X, y = preprocess_data(df)
    model, X_test, y_test, report, conf_matrix, roc_auc = train_model(X, y)

    st.subheader("Model Performance Metrics")
    st.write(f"ROC AUC Score: {roc_auc:.3f}")
    st.write("Classification Report:")
    st.json(report)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    st.subheader("Input Patient Data")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "Typical Angina",
                                  2: "Atypical Angina",
                                  3: "Non-anginal Pain",
                                  4: "Asymptomatic"
                              }[x])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1])
    rest_ecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                            format_func=lambda x: {
                                0: "Normal",
                                1: "ST-T wave abnormality",
                                2: "Left Ventricular hypertrophy"
                            }[x])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[1, 2, 3],
                         format_func=lambda x: {
                             1: "Upward",
                             2: "Flat",
                             3: "Downward"
                         }[x])

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [chest_pain],
        "trestbps": [resting_bp],
        "chol": [cholesterol],
        "fbs": [fasting_bs],
        "restecg": [rest_ecg],
        "thalach": [max_hr],
        "exang": [exercise_angina],
        "oldpeak": [oldpeak],
        "slope": [slope]
    })

    if st.button("Predict Heart Disease Risk"):
        pred_prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]
        st.write(f"### Prediction: {'Heart Disease Detected ❤️‍🩹' if pred == 1 else 'No Heart Disease Detected ✅'}")
        st.write(f"Prediction Probability: {pred_prob:.2f}")

        # Get recommendations
        rec = get_recommendations(pred, input_data.iloc[0])
        if pred == 1:
            st.write("#### Recommended Treatments:")
            st.write("**Modern Treatments:**")
            for item in rec["Modern Treatment"]:
                st.write(f"- {item}")
            st.write("**Ayurvedic Remedies:**")
            for item in rec["Ayurvedic Remedies"]:
                st.write(f"- {item}")
        else:
            st.write(rec["message"])

        plot_shap(model, X_test)


if __name__ == "__main__":
    main()
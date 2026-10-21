
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Grade Prediction Dashboard")
st.write(
    "Enter student information to estimate their final grade "
    "using a trained machine-learning regression model."
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "improved_regression_model.pkl"
)

FEATURES = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"] if isinstance(saved, dict) else saved

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()

st.info(
    "Enter values based on the student dataset. "
    "The prediction is an estimate, not a guaranteed result."
)

st.subheader("Student information")

with st.form("student_details"):
    col1, col2 = st.columns(2)
    values = {}

    field_settings = {
        "age": (15, 22, 17, "Student age (15–22)"),
        "Medu": (0, 4, 2, "Mother's education (0–4)"),
        "Fedu": (0, 4, 2, "Father's education (0–4)"),
        "traveltime": (1, 4, 2, "Travel time (1–4)"),
        "studytime": (1, 4, 2, "Weekly study time (1–4)"),
        "failures": (0, 4, 0, "Past class failures"),
        "famrel": (1, 5, 4, "Family relationship (1–5)"),
        "freetime": (1, 5, 3, "Free time (1–5)"),
        "goout": (1, 5, 3, "Going out with friends (1–5)"),
        "Dalc": (1, 5, 1, "Workday alcohol consumption (1–5)"),
        "Walc": (1, 5, 1, "Weekend alcohol consumption (1–5)"),
        "health": (1, 5, 3, "Health status (1–5)"),
        "absences": (0, 100, 0, "Number of school absences")
    }

    for index, feature in enumerate(FEATURES):
        low, high, default, label = field_settings[feature]
        with (col1 if index % 2 == 0 else col2):
            values[feature] = st.number_input(
                label,
                min_value=low,
                max_value=high,
                value=default,
                step=1,
                key=feature
            )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary"
    )

if submitted:
    try:
        input_data = pd.DataFrame([values], columns=FEATURES)

        if input_data.isnull().any().any():
            st.error("Please fill in all fields.")
        else:
            prediction = float(model.predict(input_data)[0])

            st.subheader("Prediction result")
            st.metric("Estimated final grade", f"{prediction:.2f} / 20")

            if prediction < 0 or prediction > 20:
                st.warning(
                    "The prediction is outside the usual 0–20 grade range. "
                    "Regression models can produce out-of-range estimates."
                )

            st.write("Input summary")
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Limitations: predictions depend on the training data and model quality. "
    "The estimate should not be used as the sole basis for decisions about a student."
)

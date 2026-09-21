from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Graduate Salary Predictor",
    layout="centered"
)

# -------------------------------------------------
# Load model and dropdown options once
# -------------------------------------------------
MODEL_PATH = (
    Path(__file__).parent
    / "graduate_salary_model_bundle.joblib"
)

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


model_bundle = load_model_bundle()
salary_pipeline = model_bundle["pipeline"]

# -------------------------------------------------
# App interface
# -------------------------------------------------
st.title("Graduate Salary Predictor")

st.write(
    "Select a university, school, degree, and year to estimate "
    "the gross monthly mean salary using historical survey data."
)

st.divider()

# -------------------------------------------------
# Input form
# -------------------------------------------------
with st.form("salary_prediction_form"):

    year = st.number_input(
        "Survey / prediction year",
        min_value=model_bundle["min_year"],
        max_value=model_bundle["max_year"] + 5,
        value=model_bundle["max_year"],
        step=1
    )

    university = st.selectbox(
        "University",
        options=model_bundle["universities"]
    )

    # School options update after the university is selected
    school_options = model_bundle["schools_by_university"][university]

    school = st.selectbox(
        "School",
        options=school_options
    )

    # Degree options update after university and school are selected
    degree_options = model_bundle[
        "degrees_by_university_school"
    ].get(
        (university, school),
        []
    )

    degree = st.selectbox(
        "Degree",
        options=degree_options
    )

    submitted = st.form_submit_button(
        "Predict salary",
        type="primary"
    )

# -------------------------------------------------
# Prediction result
# -------------------------------------------------
if submitted:

    new_record = pd.DataFrame({
        "year": [year],
        "university": [university],
        "school": [school],
        "degree": [degree]
    })

    predicted_salary = salary_pipeline.predict(new_record)[0]

    st.success("Prediction created.")

    st.metric(
        "Predicted Gross Monthly Mean Salary",
        f"S${predicted_salary:,.2f}"
    )

    # Warn if a future year is entered
    if year > model_bundle["max_year"]:
        st.warning(
            f"{year} is outside the model's training range "
            f"({model_bundle['min_year']}–{model_bundle['max_year']}). "
            "This prediction is an extrapolation and may be less reliable."
        )

    st.caption(
        "This is a historical programme-level estimate. "
        "It does not predict an individual graduate's salary."
    )

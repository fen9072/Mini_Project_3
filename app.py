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
# -------------------------------------------------
# Reset dependent dropdowns
# -------------------------------------------------
def reset_school_and_degree():
    st.session_state.pop("school", None)
    st.session_state.pop("degree", None)


def reset_degree():
    st.session_state.pop("degree", None)


# -------------------------------------------------
# University dropdown
# -------------------------------------------------
university = st.selectbox(
    "University",
    options=model_bundle["universities"],
    key="university",
    on_change=reset_school_and_degree
)

# Get schools only for the selected university
school_options = model_bundle["schools_by_university"].get(
    university,
    []
)

# Clear an old school choice if it does not belong to this university
if st.session_state.get("school") not in school_options:
    st.session_state.pop("school", None)

if not school_options:
    st.error("No schools are available for this university.")
    st.stop()

school = st.selectbox(
    "School",
    options=school_options,
    key="school",
    on_change=reset_degree
)

# Get degrees only for the selected university and school
degree_options = model_bundle[
    "degrees_by_university_school"
].get(
    (university, school),
    []
)

# Clear an old degree choice if it does not belong to this combination
if st.session_state.get("degree") not in degree_options:
    st.session_state.pop("degree", None)

if not degree_options:
    st.error("No degrees are available for this university and school.")
    st.stop()

degree = st.selectbox(
    "Degree",
    options=degree_options,
    key="degree"
)

# -------------------------------------------------
# Year and prediction button
# -------------------------------------------------
year = st.number_input(
    "Survey / prediction year",
    min_value=model_bundle["min_year"],
    max_value=model_bundle["max_year"] + 5,
    value=model_bundle["max_year"],
    step=1
)

if st.button("Predict salary", type="primary"):

    new_record = pd.DataFrame({
        "year": [year],
        "university": [university],
        "school": [school],
        "degree": [degree]
    })

    predicted_salary = salary_pipeline.predict(new_record)[0]

    st.metric(
        "Predicted Gross Monthly Mean Salary",
        f"S${predicted_salary:,.2f}"
    )

    if year > model_bundle["max_year"]:
        st.warning(
            f"{year} is outside the training range "
            f"({model_bundle['min_year']}–{model_bundle['max_year']}). "
            "This prediction is an extrapolation."
        )

    st.caption(
        "This is a historical programme-level estimate, "
        "not an individual graduate salary prediction."
    )


# -------------------------------------------------
# Prediction result
# -------------------------------------------------
if st.button("Predict salary", type="primary"):

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

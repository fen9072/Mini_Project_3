from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(
    page_title="Graduate Salary Predictor",
    layout="centered"
)

# -------------------------------------------------
# Load saved model bundle
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
# Functions to reset dependent dropdowns
# -------------------------------------------------
def reset_school_and_degree():
    st.session_state.pop("school_input", None)
    st.session_state.pop("degree_input", None)


def reset_degree():
    st.session_state.pop("degree_input", None)


# -------------------------------------------------
# App title
# -------------------------------------------------
st.title("Graduate Salary Predictor")

st.write(
    "Select programme information to estimate the gross "
    "monthly mean salary using historical survey data."
)

st.divider()

# -------------------------------------------------
# University dropdown
# -------------------------------------------------
university = st.selectbox(
    "University",
    options=model_bundle["universities"],
    key="university_input",
    on_change=reset_school_and_degree
)

# -------------------------------------------------
# School dropdown: filtered by university
# -------------------------------------------------
school_options = model_bundle["schools_by_university"].get(
    university,
    []
)

# Remove an old school value if it does not match the new university
if (
    "school_input" in st.session_state
    and st.session_state["school_input"] not in school_options
):
    st.session_state.pop("school_input")

if not school_options:
    st.error("No school options are available for this university.")
    st.stop()

school = st.selectbox(
    "School",
    options=school_options,
    key="school_input",
    on_change=reset_degree
)

# -------------------------------------------------
# Degree dropdown: filtered by university and school
# -------------------------------------------------
degree_options = model_bundle[
    "degrees_by_university_school"
].get(
    (university, school),
    []
)

# Remove an old degree value if it does not match the new school
if (
    "degree_input" in st.session_state
    and st.session_state["degree_input"] not in degree_options
):
    st.session_state.pop("degree_input")

if not degree_options:
    st.error(
        "No degree options are available for this "
        "university and school combination."
    )
    st.stop()

degree = st.selectbox(
    "Degree",
    options=degree_options,
    key="degree_input"
)

# -------------------------------------------------
# Year input
# -------------------------------------------------
year = st.number_input(
    "Survey / prediction year",
    min_value=model_bundle["min_year"],
    max_value=model_bundle["max_year"] + 5,
    value=model_bundle["max_year"],
    step=1,
    key="year_input"
)

# -------------------------------------------------
# Prediction button
# -------------------------------------------------
predict_clicked = st.button(
    "Predict salary",
    type="primary",
    key="predict_button"
)

if predict_clicked:

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
            f"({model_bundle['min_year']}–"
            f"{model_bundle['max_year']}). "
            "This prediction is an extrapolation and may "
            "be less reliable."
        )

    st.caption(
        "This is a historical programme-level estimate. "
        "It is not a prediction of an individual graduate's salary."
    )

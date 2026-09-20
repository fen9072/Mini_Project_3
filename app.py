from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Graduate Salary Predictor",
    page_icon="Graduation cap",
    layout="centered"
)

# Load the trained pipeline once
@st.cache_resource
def load_model():
    model_path = (
        Path(__file__).parent
        / "graduate_salary_pipeline.joblib"
    )
    return joblib.load(model_path)


salary_pipeline = load_model()

# Application interface
st.title("Graduate Salary Predictor")

st.write(
    "Estimate a programme's gross monthly mean salary using "
    "historical Graduate Employment Survey data."
)

st.divider()

year = st.number_input(
    "Survey / prediction year",
    min_value=2013,
    max_value=2030,
    value=2024,
    step=1
)

university = st.text_input(
    "University",
    value="National University of Singapore"
)

school = st.text_input(
    "School",
    value="School of Computing"
)

degree = st.text_input(
    "Degree",
    value="Computer Science"
)

if st.button("Predict salary", type="primary"):

    if not university.strip() or not school.strip() or not degree.strip():
        st.error("Please complete the university, school, and degree fields.")

    else:
        # DataFrame column names must match the training features exactly
        new_record = pd.DataFrame({
            "year": [year],
            "university": [university.strip()],
            "school": [school.strip()],
            "degree": [degree.strip()]
        })

        prediction = salary_pipeline.predict(new_record)[0]

        st.metric(
            "Predicted gross monthly mean salary",
            f"S${prediction:,.2f}"
        )

        if year < 2013 or year > 2024:
            st.warning(
                "This year is outside the model's 2013–2024 training range. "
                "The result is an extrapolation and may be less reliable."
            )

        st.caption(
            "This is a programme-level estimate based on historical Graduate "
            "Employment Survey data. It is not a prediction of an individual "
            "graduate's salary."
        )

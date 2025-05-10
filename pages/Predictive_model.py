import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import time

st.markdown("---")
st.markdown(
    "<h1 style='color: maroon;'>Insurance Charges Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")
st.write("Enter the details below to predict insurance charges:")

# Loading model and preprocessor
model = joblib.load("utils/final_model.pkl")
preprocessor = joblib.load("utils/preprocessor.pkl")

# Wrapping form in a relative container
st.markdown(
    """
    <div id="form-container" style="position: relative;">
    """,
    unsafe_allow_html=True,
)

# Data inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox(
    "Region", options=["northeast", "northwest", "southeast", "southwest"]
)

loading = st.empty()

if st.button("Predict Charges"):
    loading.markdown(
        """
        <style>
@keyframes slideDownFade {
    0% {
        opacity: 0;
        transform: translate(-50%, -100%);
    }
    100% {
        opacity: 1;
        transform: translate(-50%, -50%);
    }
}

.loader-overlay {
    position: absolute;
    top: 45%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 9999;
    background-color: rgba(255,255,255,0);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    animation: slideDownFade 0.6s ease-out;
}
</style>


<div class="loader-overlay">
    <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2Q3eWJqbmMwdzEyZHM2Ymtxenp0dHZsbDAza2JuaXlveTQ2a3QwaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/VhQFsdrgaARYr6YQIf/giphy.gif" width="120" style="display: block; margin: 0 auto;">
    <p style="margin-top: 10px; font-size: 16px; color: #333;">Processing, please wait...</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    time.sleep(3)
    loading.empty()

    # Preparig input
    input_df = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )

    try:
        # Converting and engineering features
        input_df["smoker"] = input_df["smoker"].apply(lambda x: 1 if x == "yes" else 0)

        def age_converter(a):
            if 18 <= a <= 35:
                return "Young"
            elif 36 <= a <= 55:
                return "Middle-aged"
            else:
                return "Senior"

        input_df["age_group"] = input_df["age"].apply(age_converter)
        input_df["bmi_smoker"] = input_df["bmi"] * input_df["smoker"]
        input_df["age_smoker"] = input_df["age"] * input_df["smoker"]
        input_df["age_bmi"] = input_df["age"] * input_df["bmi"]
        input_df["children_per_age"] = input_df.apply(
            lambda x: x["children"] / x["age"] if x["age"] != 0 else 0, axis=1
        )

        # Making predictions
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        st.success(f"Insurance Charges: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")


st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

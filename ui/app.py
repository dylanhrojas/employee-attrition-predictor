import os
from typing import Literal, get_args, get_origin

import requests
import streamlit as st
from pydantic import BaseModel

from schemas.employee import EmployeeInput
from src.config import THRESHOLD

st.title("Attrition Predictor")
def pydantic_form(model: type[BaseModel], key: str = "form"):
    with st.form(key=key):
        values = {}
        for name, field in model.model_fields.items():
            annotation = field.annotation
            label = field.title or name.replace("_", " ").title()

            if get_origin(annotation) is Literal:
                options = get_args(annotation)
                if len(options) == 1:
                    values[name] = options[0]
                elif len(options) == 2:
                    values[name] = st.radio(label, options)
                else:
                    values[name] = st.selectbox(label, options)

            elif annotation is int:
                ge = next((c.ge for c in field.metadata if hasattr(c, "ge")), None)
                le = next((c.le for c in field.metadata if hasattr(c, "le")), None)
                default = None if field.is_required() else field.get_default()
                if ge is not None and le is not None:
                    values[name] = st.slider(label, min_value=ge, max_value=le, value=default, step=1)
                else:
                    values[name] = st.number_input(label, value=default, step=1)

            else:
                raise NotImplementedError(f"Unsupported type for field {name}: {annotation}")

        submitted = st.form_submit_button("Predict")

    return values if submitted else None

data = pydantic_form(EmployeeInput)
if data:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=5)

    except requests.exceptions.ConnectionError:
        st.error("Can't connect with API.")
    else:
        if response.ok:
            result = response.json()
            prob = result["probability"]
            color = "red" if prob >= THRESHOLD else "green"

            st.markdown(f"""
            <style>
            div[data-testid="stMetricValue"] {{
                color: {color};
            }}
            </style>
            """, unsafe_allow_html=True)

            st.metric(label="Attrition Probability", value=f"{prob:.1%}")
        else:
            if response.status_code == 422:
                st.error(f"Error {response.status_code}: Check if all fields are filled.")
            else:
                st.error(f"Error {response.status_code}: {response.json()}")

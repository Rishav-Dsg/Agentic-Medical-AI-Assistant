import streamlit as st
from src.graph.state import MedicalState
from src.graph.medical_graph import medical_graph

uploaded = st.file_uploader(
    "Upload Chest X-Ray"
)

if uploaded:

    path = uploaded.name

    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    state: MedicalState  = {
        "image_path": path
    }

    result= medical_graph.invoke(state)

    st.write(result["report"])
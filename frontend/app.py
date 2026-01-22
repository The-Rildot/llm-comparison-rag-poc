import streamlit as st
import sys
from pathlib import Path

# allow backend imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.services.run_comparison import run_llm_comparison


st.set_page_config(
    page_title="LLM Comparison POC",
    layout="wide"
)

st.title("🔍 LLM Comparison POC")
st.write("Compare how different AI models answer the same question using identical data.")

question = st.text_input(
    "Enter your question:",
    placeholder="Can assets be deleted via the API?"
)

if st.button("Run comparison") and question:
    with st.spinner("Running models..."):
        results = run_llm_comparison(question)

    cols = st.columns(len(results))

    for col, (model, answer) in zip(cols, results.items()):
        with col:
            st.subheader(model)
            st.markdown(answer)
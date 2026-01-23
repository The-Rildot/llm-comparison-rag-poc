import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# allow backend imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.services.run_comparison import run_llm_comparison


st.set_page_config(
    page_title="LLM Comparison POC",
    layout="wide"
)

st.title("LLM Comparison POC Dashboard")
st.write("Compare how different AI models answer the same question using identical data.")

question = st.text_input(
    "Enter your question:",
    placeholder="Can assets be deleted via the API?"
)

if st.button("Run comparison") and question:
    with st.spinner("Running models..."):
        results = run_llm_comparison(question)

    cols = st.columns(len(results))

    for col, (model, data) in zip(cols, results.items()):
        with col:
            st.subheader(model)

            st.markdown("### ⏱ Metrics")
            st.json(data["metrics"])

            st.markdown("### 🧠 Response")
            st.write(data["response"])

    latency_df = pd.DataFrame([
    {
        "Model": model,
        "Latency (ms)": data["metrics"]["latency_ms"]
    }
    for model, data in results.items()
    ])

    st.subheader("Latency Comparison (ms)")
    st.bar_chart(latency_df.set_index("Model"))


    tps_df = pd.DataFrame([
        {
            "Model": model,
            "Tokens/sec": data["metrics"]["tokens_per_second"]
        }
        for model, data in results.items()
    ])
    st.subheader("Throughput Comparison (tokens/sec)")
    st.bar_chart(tps_df.set_index("Model"))
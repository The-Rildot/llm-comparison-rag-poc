import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Allow backend imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.services.run_comparison import run_llm_comparison

# Page setup
st.set_page_config(
    page_title="LLM Comparison POC",
    layout="wide"
)

st.title("LLM Comparison POC Dashboard")
st.write("Compare how different AI models answer the same question using identical data.")

# User inputs
question = st.text_input(
    "Enter your question:",
    placeholder="Can assets be deleted via the API?"
)

col1, col2 = st.columns(2)

with col1:
    run_type = st.radio(
        "Run type",
        ["Single response", "Multiple responses"]
    )

with col2:
    stream_mode = st.toggle(
        "Streaming output",
        value=False
    )

# Map UI → backend arguments
mode = "single" if run_type == "Single response" else "multiple"

# streaming disabled automatically for multi-run
stream = stream_mode if mode == "single" else False

# Run comparison
if st.button("Run comparison") and question:

    with st.spinner("Running models..."):
        results = run_llm_comparison(
            question=question,
            mode=mode,
            stream=stream
        )

    # Display model outputs
    cols = st.columns(len(results))

    for col, (model, data) in zip(cols, results.items()):
        with col:

            st.subheader(model)

            # SINGLE RESPONSE MODE
            if mode == "single":

                st.markdown("### Response")

                if stream:
                    placeholder = st.empty()
                    full_text = ""

                    for token in data["response"]:
                        full_text += token
                        placeholder.markdown(full_text)

                else:
                    st.write(data["response"])

                st.markdown("### Metrics")
                st.json({
                    "latency_ms": data["latency_ms"]
                })

            # MULTI-RUN MODE
            else:
                st.markdown("### Final Response")
                st.write(data["response"])

                st.markdown("### Metrics")
                st.json(data["metrics"])


    # Charts (only when metrics exist)
    if mode == "single":

        latency_df = pd.DataFrame([
            {
                "Model": model,
                "Latency (ms)": data["latency_ms"]
            }
            for model, data in results.items()
        ])

        st.subheader("Latency Comparison (ms)")
        st.bar_chart(latency_df.set_index("Model"))

    else:
        latency_df = pd.DataFrame([
            {
                "Model": model,
                "p50 (ms)": data["metrics"]["latency_p50_ms"],
                "p95 (ms)": data["metrics"]["latency_p95_ms"]
            }
            for model, data in results.items()
        ])

        st.subheader("Latency Distribution Comparison")
        st.bar_chart(latency_df.set_index("Model"))
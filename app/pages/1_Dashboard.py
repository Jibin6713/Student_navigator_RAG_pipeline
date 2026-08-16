import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from monitoring.dashboard_data import (
    load_feedback,
    load_llm_eval_results,
    load_retrieval_eval_results,
)

st.set_page_config(page_title="Dashboard — UoA Student Navigator", page_icon="📊", layout="wide")

st.title("📊 Monitoring Dashboard")

# --- live user feedback ---
st.header("Live user feedback")

feedback = load_feedback()

if not feedback:
    st.info(
        "No feedback collected yet — ask questions and rate answers in the "
        "main app to populate this section."
    )
else:
    df = pd.DataFrame(feedback)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["rating_label"] = df["rating"].map({1: "👍 Positive", -1: "👎 Negative"})

    col1, col2, col3 = st.columns(3)
    col1.metric("Total feedback", len(df))
    col2.metric("Positive", int((df["rating"] == 1).sum()))
    col3.metric("Negative", int((df["rating"] == -1).sum()))

    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(df, names="rating_label", title="Feedback ratio")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        daily = (
            df.groupby([df["created_at"].dt.date, "rating_label"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            daily, x="created_at", y="count", color="rating_label",
            title="Feedback over time", barmode="stack",
        )
        st.plotly_chart(fig, use_container_width=True)

    by_variant = df.groupby(["variant", "rating_label"]).size().reset_index(name="count")
    fig = px.bar(
        by_variant, x="variant", y="count", color="rating_label",
        title="Feedback by prompt variant", barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent feedback")
    st.dataframe(
        df[["created_at", "question", "rating_label", "method", "variant"]]
        .sort_values("created_at", ascending=False)
        .head(20),
        use_container_width=True,
    )

# --- offline evaluation results ---
st.header("Offline evaluation results")

retrieval_results = load_retrieval_eval_results()

if retrieval_results:
    rdf = pd.DataFrame(retrieval_results)
    rdf_melted = rdf.melt(
        id_vars="method", value_vars=["hit_rate", "mrr"],
        var_name="metric", value_name="score",
    )
    fig = px.bar(
        rdf_melted, x="method", y="score", color="metric", barmode="group",
        title="Retrieval evaluation: hit rate & MRR by method",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No retrieval evaluation results found — run `python -m eval.retrieval_eval`.")

llm_results = load_llm_eval_results()

if llm_results:
    ldf = pd.DataFrame(llm_results)

    rel_counts = ldf.groupby(["variant", "relevance"]).size().reset_index(name="count")
    fig = px.bar(
        rel_counts, x="variant", y="count", color="relevance", barmode="stack",
        title="LLM evaluation: relevance by prompt variant",
    )
    st.plotly_chart(fig, use_container_width=True)

    faith_counts = ldf.groupby(["variant", "faithfulness"]).size().reset_index(name="count")
    fig = px.bar(
        faith_counts, x="variant", y="count", color="faithfulness", barmode="stack",
        title="LLM evaluation: faithfulness by prompt variant",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No LLM evaluation results found — run `python -m eval.llm_eval`.")

"""Streamlit app: score an English comment and show a moderation recommendation."""

import pandas as pd
import streamlit as st

from portfolio_projekt.recommend import explain, load_bundle, recommend, score

st.set_page_config(page_title="Toxic Comment Checker", page_icon="🛡️")
st.title("🛡️ Toxic Comment Checker")
st.caption(
    "Prototype for English comments · TF-IDF + logistic regression · "
    "The software recommends, a human decides."
)

text = st.text_area("Comment", placeholder="Type or paste an English comment ...", height=120)

if st.button("Analyze", type="primary", disabled=not text.strip()):
    thresholds = load_bundle()["thresholds"]
    scores = score(text)
    result = recommend(scores, thresholds)

    # Recommendation, colored by severity
    st.subheader("Recommendation")
    message = f"**Level {result['level']}: {result['action']}**  \n{result['reason']}"
    if result["level"] == 0:
        st.success(message)
    elif result["level"] <= 2:
        st.warning(message)
    else:
        st.error(message)

    # Overall toxicity score
    st.metric("Toxicity score", f"{scores['toxicity']:.2f}")
    st.progress(scores["toxicity"])

    # All categories
    st.subheader("Categories")
    df_scores = pd.DataFrame({
        "category": list(scores),
        "probability": [round(value, 2) for value in scores.values()],
        "flagged": ["⚠️" if scores[label] >= thresholds[label] else "" for label in scores],
    })
    st.dataframe(df_scores, hide_index=True)

    # Explanation
    words = explain(text)
    if words:
        st.subheader("Why?")
        st.write("Words that push the toxicity score up the most:")
        st.write(", ".join(f"**{word}** ({value})" for word, value in words))

    # Actions as disabled previews
    st.subheader("Actions")
    col_hide, col_delete, col_ok = st.columns(3)
    col_hide.button("Hide", disabled=True)
    col_delete.button("Delete", disabled=True)
    col_ok.button("Mark as OK", disabled=True)
    st.caption("Demo: actions are disabled in this prototype.")

# Batch check as a disabled preview
st.divider()
st.subheader("Batch check (coming soon)")
st.file_uploader("Upload a CSV file with comments", type="csv", disabled=True)
st.caption("Planned: check many comments at once and sort them by toxicity.")
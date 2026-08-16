import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from indexes.db import get_connection
from monitoring.feedback_store import init_feedback_table, save_feedback
from rag.pipeline import answer

st.set_page_config(page_title="UoA Student Navigator", page_icon="🎓")

st.title("🎓 UoA Student Navigator")
st.caption(
    "Ask about study options, fees, entry requirements, student life, or "
    "campus services at the University of Auckland."
)

# fresh connection each rerun — Streamlit doesn't guarantee reruns stay on
# the same thread, and sqlite3 connections can't cross threads
conn = get_connection()
init_feedback_table(conn)

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(f"- [{s['title']} — {s['heading']}]({s['url']})")

            if message.get("feedback") is None:
                col1, col2, _ = st.columns([1, 1, 8])

                if col1.button("👍", key=f"up_{i}"):
                    save_feedback(conn, message["question"], message["content"], rating=1)
                    message["feedback"] = 1
                    st.rerun()

                if col2.button("👎", key=f"down_{i}"):
                    save_feedback(conn, message["question"], message["content"], rating=-1)
                    message["feedback"] = -1
                    st.rerun()
            else:
                st.caption("Thanks for the feedback!")

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Searching the University of Auckland website..."):
        result = answer(question, conn=conn)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "question": question,
        "sources": result["sources"],
        "feedback": None,
    })

    st.rerun()

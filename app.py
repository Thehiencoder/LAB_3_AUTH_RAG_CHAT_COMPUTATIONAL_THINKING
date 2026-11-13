"""Practice 03: Sign-in Travel Assistant with RAG Chat (Lumen Travel).

Combines:
- Lab 03: Streamlit UI + Firebase Email/Password Authentication + st.session_state management.
- Lab 04 / Lab 05: Ollama embeddinggemma + Cosine Similarity Top-k retrieval + qwen3:4b grounded generation.
"""

import time
import streamlit as st

from config import CHAT_MODEL, EMBED_MODEL, MIN_SIMILARITY_SCORE, OLLAMA_BASE_URL, TOP_K
from firebase_client import FirebaseError, create_account, sign_in, validate_config
from rag import (
    RAGError,
    answer_question,
    build_index,
    citation_check,
    load_knowledge_base,
    validate_ollama_config,
)

st.set_page_config(
    page_title="Lumen Travel Assistant",
    page_icon="🧭",
    layout="centered",
)


def forget_user() -> None:
    """Clear all session state values belonging to ONE visitor, including chat history."""
    for key in (
        "user",
        "messages",
        "auth_email",
        "auth_password",
        "notice",
        "auth_error",
        "clear_password",
        "account_created",
    ):
        st.session_state.pop(key, None)


@st.cache_resource(show_spinner="Embedding travel guide with embeddinggemma...")
def get_indexed_guide(ollama_url: str, embed_model: str):
    """Embed knowledge_base.json ONCE across sessions and reruns (never on every question)."""
    records = load_knowledge_base()
    return build_index(records)


st.title("🧭 Lumen Travel Assistant")
st.caption(
    "Ho Chi Minh City Travel Guide · Grounded RAG Chat (`embeddinggemma` + `qwen3:4b`) protected by Firebase Sign-In"
)

try:
    validate_config()
except FirebaseError as error:
    st.info(str(error))
    st.stop()

# Reset credentials BEFORE drawing their widgets, including on failed sign-in (Lab 03 pattern).
if st.session_state.pop("clear_password", False):
    st.session_state["auth_password"] = ""
if st.session_state.pop("account_created", False):
    st.success("Account created. Sign in with the same app credentials.")
if "auth_error" in st.session_state:
    st.error(st.session_state.pop("auth_error"))
if "notice" in st.session_state:
    st.info(st.session_state.pop("notice"))

try:
    if "user" not in st.session_state:
        st.subheader("Sign in to chat with the Travel Assistant")
        st.write(
            "Only signed-in users may chat. Use a separate lab-app password, not your Google password."
        )
        with st.form("account_form", clear_on_submit=True):
            email = st.text_input("Email", key="auth_email")
            password = st.text_input("Password", type="password", key="auth_password")
            col1, col2 = st.columns(2)
            with col1:
                login = st.form_submit_button("Sign in", use_container_width=True)
            with col2:
                register = st.form_submit_button("Create account", use_container_width=True)

        # Forms only submit on a button click, not after every keystroke.
        if register:
            create_account(email, password)
            st.session_state["account_created"] = True
            st.session_state["clear_password"] = True
            st.rerun()
        elif login:
            user = sign_in(email, password)
            forget_user()
            st.session_state["user"] = user
            st.session_state["messages"] = []
            st.rerun()  # Show the signed-in chat page.

    else:
        user = st.session_state["user"]

        # Expire session automatically if token lifetime elapsed
        if time.time() >= user.get("expires_at", 0):
            forget_user()
            st.session_state["notice"] = "Your session has expired. Please sign in again."
            st.rerun()

        # Sidebar: User profile, RAG settings, and Sign out button
        with st.sidebar:
            st.subheader("👤 Signed-in User")
            st.write(f"**Email:** {user['email']}")
            st.caption(f"UID: `{user['uid']}`")
            if st.button("Sign out", use_container_width=True):
                forget_user()
                st.rerun()

            st.divider()
            st.subheader("⚙️ RAG Settings")
            k_val = st.slider(
                "Top-k retrieved entries (k)",
                min_value=1,
                max_value=5,
                value=TOP_K,
                help="Number of guide entries retrieved by cosine similarity.",
            )
            st.caption(f"Embedding model: `{EMBED_MODEL}`")
            st.caption(f"Chat model: `{CHAT_MODEL}`")

            if st.button("Clear chat history", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

        # Validate Ollama configuration and embed the guide ONCE via @st.cache_resource
        try:
            validate_ollama_config()
            indexed_docs, doc_vectors = get_indexed_guide(OLLAMA_BASE_URL, EMBED_MODEL)
        except RAGError as rag_err:
            st.error(f"**Ollama Server Error:** {rag_err}")
            if st.button("Retry connection"):
                st.cache_resource.clear()
                st.rerun()
            st.stop()

        with st.expander(f"📖 Browse Ho Chi Minh City Travel Guide ({len(indexed_docs)} entries)"):
            for doc in indexed_docs:
                tags_str = ", ".join(doc.get("tags", []))
                st.markdown(
                    f"**[{doc['id']}] {doc['title']}** (`{doc.get('category', 'place')}` · tags: `{tags_str}`)  \n"
                    f"{doc['text']}  \n"
                    f"*Source: {doc['source']}*"
                )
                st.divider()

        # Session-only chat history (kept in st.session_state, cleared at sign-out, no database)
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display conversation history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("hits"):
                    with st.expander("🔍 Retrieved guide entries & Cosine similarity scores"):
                        for hit in msg["hits"]:
                            st.markdown(
                                f"- **`[{hit['id']}]` {hit['title']}** — Score: `{hit['score']:.4f}`  \n"
                                f"  {hit['text']}  \n"
                                f"  *Category: `{hit.get('category', 'N/A')}` | Tags: `{', '.join(hit.get('tags', []))}` | Source: {hit['source']}*"
                            )
                        if msg.get("citations"):
                            cited = ", ".join(msg["citations"].get("cited_ids", [])) or "None"
                            st.caption(f"Cited entry IDs: {cited}")

        # Question submission form (explicitly handles empty questions as required)
        with st.form("question_form", clear_on_submit=True):
            col_input, col_submit = st.columns([5, 1])
            with col_input:
                question = st.text_input(
                    "Ask a question about Ho Chi Minh City",
                    placeholder="Ask about opening hours, ticket prices, or places in the guide...",
                    label_visibility="collapsed",
                )
            with col_submit:
                ask_clicked = st.form_submit_button("Send", use_container_width=True)

        if ask_clicked:
            if not question or not question.strip():
                st.warning("Please enter a non-empty question before clicking Send.")
            else:
                clean_q = question.strip()
                st.session_state["messages"].append({"role": "user", "content": clean_q})

                with st.chat_message("user"):
                    st.markdown(clean_q)

                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Retrieving top-k guide entries and generating answer..."):
                            answer, hits = answer_question(
                                clean_q,
                                indexed_docs,
                                doc_vectors,
                                k=k_val,
                                min_score=MIN_SIMILARITY_SCORE,
                            )
                            citations = citation_check(answer, hits)

                        st.markdown(answer)
                        with st.expander(
                            "🔍 Retrieved guide entries & Cosine similarity scores",
                            expanded=True,
                        ):
                            for hit in hits:
                                st.markdown(
                                    f"- **`[{hit['id']}]` {hit['title']}** — Score: `{hit['score']:.4f}`  \n"
                                    f"  {hit['text']}  \n"
                                    f"  *Category: `{hit.get('category', 'N/A')}` | Tags: `{', '.join(hit.get('tags', []))}` | Source: {hit['source']}*"
                                )
                            cited = ", ".join(citations.get("cited_ids", [])) or "None"
                            st.caption(f"Cited entry IDs: {cited}")

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "hits": hits,
                                "citations": citations,
                            }
                        )
                    except RAGError as rag_err:
                        st.error(str(rag_err))

except FirebaseError as error:
    if error.sign_in_again:
        forget_user()
        st.session_state["notice"] = str(error)
        st.rerun()
    if "user" not in st.session_state:
        st.session_state["clear_password"] = True
        st.session_state["auth_error"] = str(error)
        st.rerun()
    st.error(str(error))

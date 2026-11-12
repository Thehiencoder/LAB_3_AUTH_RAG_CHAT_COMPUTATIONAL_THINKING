
import time
import streamlit as st

from firebase_client import FirebaseError, create_account, sign_in, validate_config

st.set_page_config(
    page_title="Lumen Travel Assistant",
    page_icon="🧭",
    layout="centered",
)


def forget_user() -> None:
    """Clear all session state values belonging to the current visitor."""
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


st.title("🧭 Lumen Travel Assistant")
st.caption("Firebase Authentication protects access. Only signed-in users may chat.")

try:
    validate_config()
except FirebaseError as error:
    st.info(str(error))
    st.stop()

# Reset credentials BEFORE drawing their widgets, including on failed sign-in.
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
        st.write("Use a separate lab-app password, not your Google password.")
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

        # Automatically expire session if token has expired
        if time.time() >= user.get("expires_at", 0):
            forget_user()
            st.session_state["notice"] = "Your session has expired. Please sign in again."
            st.rerun()

        with st.sidebar:
            st.subheader("👤 Signed-in User")
            st.write(f"**Email:** {user['email']}")
            st.caption(f"UID: `{user['uid']}`")
            if st.button("Sign out", use_container_width=True):
                forget_user()
                st.rerun()

        st.subheader("💬 Travel Guide RAG Chat")
        st.info("You are signed in!")

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


import streamlit as st
from auth import AzureADOAuthProvider
from streamlit_local_storage import LocalStorage
from auth import check_user_eligibility

# Initialize LocalStorage object
local_storage = LocalStorage(key="login_data")

auth_provider = AzureADOAuthProvider()

# Session and local storage check
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "token" not in st.session_state:
    st.session_state.token = None
if "logout_clicked" not in st.session_state:
    st.session_state.logout_clicked = False

# Check local storage for login persistence
stored_login_status = local_storage.getItem("logged_in")
stored_token = local_storage.getItem("token")

if stored_login_status == "true" and stored_token:
    st.session_state.logged_in = True
    st.session_state.token = stored_token

st.header("Welcome to Streamlit")
authorize_url = f"{auth_provider.get_authorize_url()}&prompt=login"

# Handle login if the authorization code is in query params and user is not logged in
if "code" in st.query_params and not st.session_state.logged_in:
    code = st.query_params.get("code")
    try:
        st.session_state.token = auth_provider.get_token(code)
        st.query_params.clear()

        user_info = auth_provider.get_user_info(st.session_state.token)

        # List of authorized users
        login_users_list = [
            "smaddali@nvidia.com"
        ]
        if user_info['mail'] in login_users_list:
            st.session_state.logged_in = True
            local_storage.setItem("logged_in", "true", key="login_status_key")
            local_storage.setItem("token", st.session_state.token, key="token_key")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display logged-in status and user info
if st.session_state.logged_in:

    user_info = auth_provider.get_user_info(st.session_state.token)
    dl_user_allowed = check_user_eligibility(
        user_info=user_info,
        dls_list=[""],
    )
    if dl_user_allowed:
        st.write("You are logged in.")
        st.write(user_info) 
    else:
        st.write("Please request access to the application using the DL address provided. Thank you.")

    if st.button("Logout", key="logout_button"):
        # Clear session state and local storage on logout
        for key in st.session_state.keys():
            del st.session_state[key]
        local_storage.deleteAll()
        st.session_state.logout_clicked = True  # Set logout flag
        st.rerun()  # First rerun to clear session and storage

# Extra check to enforce logout on rerun
if st.session_state.get("logout_clicked"):
    st.session_state.logout_clicked = False  # Reset the flag
    st.rerun()  # Second rerun to refresh the interface fully

# Show login button if the user is not logged in
elif not st.session_state.logged_in:
    if st.button("Login to Nvidia Account", key="login_button"):
        st.write("Redirecting to Microsoft login...")
        st.markdown(f"<meta http-equiv='refresh' content='0;url={authorize_url}'>", unsafe_allow_html=True)






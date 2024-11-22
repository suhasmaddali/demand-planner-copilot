import streamlit as st

class StreamlitCredentials:

    def __init__(self):
        # Getting access to credentials for loading data to database
        self.platform = st.secrets['DATABASE']['PLATFORM']
        self.aws_access_key_id = st.secrets['DATABASE']['AWS_ACCESS_KEY_ID']
        self.aws_secret_access_key = st.secrets['DATABASE']['AWS_SECRET_ACCESS_KEY']
        self.region_name = st.secrets['DATABASE']['REGION_NAME']
        self.memory_location = st.secrets['DATABASE']['BUCKET']
        self.memory_location_files = st.secrets['DATABASE']['BUCKET_FILES']
        self.number = st.secrets['DATABASE']['NUMBER']

        # Getting access to user credentials to validate login
        self.username_credentials = st.secrets['USER CREDENTIALS']['USERNAME']
        self.password_credentials = st.secrets['USER CREDENTIALS']['PASSWORD']

        self.base_url = st.secrets['USER CREDENTIALS']['BASE_URL']
        self.api_key = st.secrets['USER CREDENTIALS']['API_KEY']
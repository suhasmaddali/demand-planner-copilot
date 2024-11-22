
from openai import OpenAI
import streamlit as st
import re
import traceback
import pandas as pd
import numpy as np
import time
from src.debugger import Debugger
from src.recommender import Recommender
from src import debugger
from datetime import datetime
import os
from dotenv import load_dotenv
from PIL import Image
import boto3
import uuid
from io import StringIO
import time
from langchain_nvidia_ai_endpoints import register_model, Model, ChatNVIDIA
import glob
from auth import AzureADOAuthProvider
from src.prompts import system_message_analyze, system_message_compare_changes
# from streamlit_local_storage import LocalStorage
from datetime import date
import plotly.graph_objects as go
from auth import check_user_eligibility
from src.utils import index_time_profile, convert_to_anaplan

import streamlit as st

# # Getting access to credentials for loading data to database
# platform = st.secrets['DATABASE']['PLATFORM']
# aws_access_key_id = st.secrets['DATABASE']['AWS_ACCESS_KEY_ID']
# aws_secret_access_key = st.secrets['DATABASE']['AWS_SECRET_ACCESS_KEY']
# region_name = st.secrets['DATABASE']['REGION_NAME']
# memory_location = st.secrets['DATABASE']['BUCKET']
# number = st.secrets['DATABASE']['NUMBER']

# # Getting access to user credentials to validate login
# username_credentials = st.secrets['USER CREDENTIALS']['USERNAME']
# password_credentials = st.secrets['USER CREDENTIALS']['PASSWORD']

# base_url = st.secrets['USER CREDENTIALS']['BASE_URL']
# api_key = st.secrets['USER CREDENTIALS']['API_KEY']



# Getting access to NVCF credentials for inference
# NVCF_CHAT_FUNCTION_ID = st.secrets['NVCF CREDENTIALS']['NVCF_CHAT_FUNCTION_ID']
# NVCF_URL = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{NVCF_CHAT_FUNCTION_ID}"
# NVCF_API_KEY = st.secrets['NVCF CREDENTIALS']['NVCF_API_KEY']
# MODEL = "meta/llama-3.1-8b-instruct"
# os.environ['NVIDIA_API_KEY'] = NVCF_API_KEY

import os

# Getting access to credentials for loading data to the database
platform = os.getenv('PLATFORM')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('REGION_NAME')
memory_location = os.getenv('BUCKET')
number = os.getenv('NUMBER')

# Getting access to user credentials to validate login
username_credentials = os.getenv('USERNAME')
password_credentials = os.getenv('PASSWORD')

base_url = os.getenv('BASE_URL')
api_key = os.getenv('API_KEY')

NVCF_CHAT_FUNCTION_ID = os.getenv('NVCF_CHAT_FUNCTION_ID')
NVCF_URL = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{NVCF_CHAT_FUNCTION_ID}"
NVCF_API_KEY = os.getenv('NVCF_API_KEY')
MODEL = "meta/llama-3.1-8b-instruct"
os.environ['NVIDIA_API_KEY'] = NVCF_API_KEY

# Initialize S3 client
s3 = boto3.client(
    platform,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

def aws_credentials_feedback():
    # AWS S3 Bucket details
    bucket_name = memory_location
    s3_file_path = 'feedback/feedback_data.xlsx'  # Path in the S3 bucket
    local_file_path = 'feedback_data.xlsx'

    return bucket_name, s3_file_path, local_file_path, s3


def aws_credentials_chat_history():
    # AWS S3 Bucket details
    bucket_name = memory_location
    s3_file_path = 'saved_chats/chat_history.xlsx'  # Path in the S3 bucket
    local_file_path = 'chat_history.xlsx'

    return bucket_name, s3_file_path, local_file_path, s3

@ st.cache_data()
def get_aws_files():

    # Initialize S3 client
    s3 = boto3.client(
        platform,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
        
    # Use the list_objects_v2 API to get objects in the bucket
    result = s3.list_objects_v2(Bucket=memory_location, Prefix='DP Allocation File/')
    extracted_values = [os.path.basename(d['Key']) for d in result['Contents']]

    return extracted_values[1:]

@ st.cache_data()
def get_local_files():

    data_path = "../customer_allocation_data_masked"

    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    return csv_files

@ st.cache_data()
def get_current_fiscal_dates():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data/Time Profile for BOT.csv")

    df_time_profile = pd.read_csv(file_path)
    df_time_profile['Day'] = pd.to_datetime(df_time_profile['Day'], format='%m/%d/%Y')

    current_date = datetime.now()
    formatted_date = current_date.strftime('%m/%d/%Y')
    current_date = pd.to_datetime(formatted_date, format='%m/%d/%Y')

    fiscal_info = df_time_profile[df_time_profile['Day'] == current_date]

    fiscal_month = fiscal_info['Fiscal Month'].values[0]
    fiscal_quarter = fiscal_info['Fiscal Quarter'].values[0]
    fiscal_year = fiscal_info['Fiscal Year'].values[0]

    return fiscal_month, fiscal_quarter, fiscal_year

bucket_name, s3_file_path_feedback, local_file_path_feedback, s3 = aws_credentials_feedback()
_, s3_file_path_chat_history, local_file_path_chat_history, _ = aws_credentials_chat_history()

# Encapsulated function to handle saving feedback locally
def save_feedback_locally(feedback, file_path):
    current_time = pd.Timestamp.now()
    feedback_data = pd.DataFrame(
        {
            'SessionID': [st.session_state['session_id']],
            'Timestamp': [current_time], 
            'Feedback': [feedback]
        }
    )
    
    if os.path.exists(file_path):
        existing_data = pd.read_excel(file_path)
        feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)
    
    feedback_data.to_excel(file_path, index=False)

def save_chat_history_locally(data_list, file_path):
    # Capture the current session ID and timestamp
    session_id = st.session_state.get('session_id', 'default_session')
    current_time = pd.Timestamp.now()
    
    # Convert the list of dictionaries to a DataFrame
    data_df = pd.DataFrame(data_list)
    
    # Add SessionID and Timestamp columns
    data_df.insert(0, 'Timestamp', current_time)  # Insert Timestamp as the first column
    data_df.insert(0, 'SessionID', session_id)    # Insert SessionID as the first column

    # If the file already exists, read the existing data and concatenate it with new data
    if os.path.exists(file_path):
        existing_data = pd.read_excel(file_path)
        data_df = pd.concat([existing_data, data_df], ignore_index=True)

    # Save the updated data back to the Excel file
    data_df.to_excel(file_path, index=False)

# Encapsulated function to upload feedback to S3
def upload_feedback_to_s3(s3_object, file_path, bucket, s3_key):
    try:
        s3_object.upload_file(file_path, bucket, s3_key)
        # st.success(f":material/thumb_up: Thank you for the feedback")
    except Exception as e:
        st.error(f"Error uploading file: {e}")

# Encapsulated function to upload feedback to S3
def upload_chat_history_to_s3(s3_object, file_path, bucket, s3_key):
    try:
        s3_object.upload_file(file_path, bucket, s3_key)
    except Exception as e:
        st.error(f"Error uploading file: {e}")

def extract_code(response_text: str):

    code_blocks = re.findall(r"```(python)?(.*?)```", response_text, re.DOTALL)
    code = "\n".join([block[1].strip() for block in code_blocks])

    if code:
        with open("llm_generated_code.py", "w") as file:
            file.write(code)
    return code

def stream_responses(input_stream):

    response_list = []
    botmsg = st.empty()
    for chunk in input_stream:
        text = chunk.choices[0].delta.content
        if text:
            response_list.append(text)
            result = "".join(response_list).strip()
            botmsg.write(result + "▌")
            time.sleep(0.05)
    if result:
        botmsg.write(result)  
    return result

def stream_responses_static(input_stream):
    response_list = []
    botmsg = st.empty()
    for chunk in input_stream.split(' '):
        response_list.append(chunk)
        result = " ".join(response_list).strip()
        botmsg.write(result + "▌")
        time.sleep(0.1)
    botmsg.write(result)
    return result

threshold = 0

def color_code(val):
    if isinstance(val, (int, float)):
        if val > threshold:
            color = 'green'
            return f'color: {color}'
        elif val < threshold:
            color = 'red'
            return f'color: {color}'
        else:
            return ''  
    else:
        return ''  
    
@st.dialog("Anaplan View", width="large")
def anaplan_view(data):
    st.write("Anaplan View")
    st.dataframe(data)

@st.dialog("Anaplan View", width="large")
def anaplan_view_compare_changes(data1, data2):
    st.write("Anaplan View")
    st.dataframe(data1)
    st.dataframe(data2)

@st.cache_data
def load_data(file, sheet_name):
    return pd.read_excel(file, sheet_name=sheet_name)

@st.cache_data
def loading_csv(file):
    response = s3.get_object(Bucket=bucket_name, Key=f'DP Allocation File/{file}')
    csv_content = response['Body'].read().decode('utf-8')
    output = pd.read_csv(StringIO(csv_content))

    return output

@st.cache_data
def loading_local_csv(file):
    data_path = '../customer_allocation_data_masked'
    file_path = data_path + f"/{file}"
    file_loaded = pd.read_csv(file_path, encoding='ISO-8859-1')
    return file_loaded


@st.cache_data
def divide_data(data, divider):
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_features] = data[numerical_features] / divider
    return data

@st.cache_data
def load_excel_file(file):
    excel_file = pd.read_excel(file)
    return excel_file

@st.cache_data
def get_sheet_columns(dfs):
    return {sheet: df.columns.tolist() for sheet, df in dfs.items()}

@st.cache_data
def get_categorical_columns(dfs):

    datetime_columns = ['Ord Date', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date', 'CRD Date', 'CRD Date Adjusted']
    categorical_columns = {}
    for col in dfs.columns:
        if col not in datetime_columns:
            if dfs[col].dtype == "object":
                categorical_columns[col] = list(dfs[col].unique())
    return categorical_columns

@st.cache_data
def compare_categorical_columns(df1, df2):
    # Specify the datetime columns to exclude
    datetime_columns = ['Ord Date', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date', 'CRD Date', 'CRD Date Adjusted']
    
    # Initialize dictionaries to hold unique values
    df1_unique = {}  # Values in df1 but not in df2
    df2_unique = {}  # Values in df2 but not in df1
    intersection = {}  # Intersection of values between df1 and df2
    
    # Get the columns common to both dataframes and exclude datetime columns
    common_columns = [col for col in df1.columns if col in df2.columns and col not in datetime_columns]
    
    # Iterate over the common columns and compare the unique values
    for col in common_columns:
        if df1[col].dtype == "object" and df2[col].dtype == "object":  # Check if the column is categorical
            unique_values_df1 = set(df1[col].unique())  # Get unique values for the column in df1
            unique_values_df2 = set(df2[col].unique())  # Get unique values for the column in df2

            # Calculate differences and intersections
            df1_unique[col] = list(unique_values_df1 - unique_values_df2)  # Values in df1 but not in df2
            df2_unique[col] = list(unique_values_df2 - unique_values_df1)  # Values in df2 but not in df1
            intersection[col] = list(unique_values_df1 & unique_values_df2)  # Intersection of the two

    return df1_unique, df2_unique, intersection

@st.cache_data
def get_numerical_columns(dfs):

    numerical_columns = []
    for col in dfs.columns:
        if np.issubdtype(dfs[col].dtype, np.integer) or np.issubdtype(dfs[col].dtype, np.floating):
            numerical_columns.append(col)

    return numerical_columns

@st.cache_data
def get_datetime_columns(dfs):
    datetime_columns = []
    actual_datetime_columns = ['Ord Date', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date', 'CRD Date', 'CRD Date Adjusted']
    for col in dfs.columns:
        if col in actual_datetime_columns:
            datetime_columns.append(col)
        # if dfs[col].dtype == "<M8[ns]":
        #     datetime_columns.append(col)
    return datetime_columns

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'task_submitted' not in st.session_state:
    st.session_state['task_submitted'] = False

if 'task_selected_state' not in st.session_state:
    st.session_state['task_selected_state'] = None

if 'display_colored_state' not in st.session_state:
    st.session_state['display_colored_state'] = None

if 'share_chat_history_button' not in st.session_state:
    st.session_state['share_chat_history_button'] = None

def content_screen():

    logo = Image.open("images/nvidia-logo-horz (1).png")

    st.sidebar.image(logo, width=150)

    load_dotenv()

    if 'messages_display' not in st.session_state:
        st.session_state.messages_display = []

    if 'debugger_messages' not in st.session_state:
        st.session_state.debugger_messages = []

    if 'old_formatted_date_time' not in st.session_state:
        now = datetime.now()
        st.session_state.old_formatted_date_time = now.strftime("%B %d, %Y %I:%M:%S %p")
    
    if 'new_formatted_date_time' not in st.session_state:
        st.session_state.new_formatted_date_time = None

    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
        st.session_state.conversations[f'{st.session_state.old_formatted_date_time}'] = []

    if 'flag_new_chat' not in st.session_state:
        st.session_state.flag_new_chat = 0

    if 'conversation_key' not in st.session_state:
        st.session_state.conversation_key = st.session_state.old_formatted_date_time

    if 'button_clicked_flag' not in st.session_state:
        st.session_state.button_clicked_flag = False

    if 'conversation_selected' not in st.session_state:
        st.session_state.conversation_selected = None

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())

    if 'clear_chat_history_button' not in st.session_state:
        st.session_state.clear_chat_history_button = False

    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = []

    if "debugger" not in st.session_state:
        st.session_state["debugger"] = []

    if "recommender" not in st.session_state:
        st.session_state["recommender"] = []

    # Clearing the LLM generated code 
    with open("llm_generated_code.py", "w") as file:
        pass
                    
    # Inject CSS to change title color
    st.markdown("""
        <style>
        .title {
            color: #76B900;
            font-size: 2.5em;
        }
        </style>
        """, unsafe_allow_html=True)

    tab_selected = st.sidebar.tabs(["Configuration", "Chat History", "Feedback"])

    with tab_selected[0]:

        private_nvcf_endpoints = st.toggle("Private NVCF Endpoints")
        help_box = """
        Analyze single data source - Analyzes a single snapshot file for insights and advanced analytics.\n
        Compare changes between data sources - Compares two snapshot files for changes, insights and advanced analytics.
        """
        task_selected = st.radio(":material/settings: Select the task:", ["Analyze single data source", "Compare changes between data sources"],
                                                  help=help_box)
        generate_prompt_suggestions = st.toggle("Generate Prompt Suggestions")
        st.session_state['task_selected_state'] = task_selected
        if not private_nvcf_endpoints:
            st.session_state["llm_model"] = "meta/llama-3.1-70b-instruct"
            st.session_state["debugger"] = "meta/llama-3.1-70b-instruct"
            st.session_state["recommender"] = "meta/llama-3.1-70b-instruct"

            client = OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
        else:
            register_model(Model(
                id=MODEL, 
                model_type="chat", 
                client="ChatNVIDIA", 
                endpoint=NVCF_URL
                )
            )
            client = ChatNVIDIA(
                model=MODEL,
                temperature=0.0,
                max_tokens=4096
            )
            st.session_state["llm_model"] = MODEL

        task_selected = "Analyze single data source"
        column_width = st.slider(":material/settings: Adjust displayed column width:", 100, 1000, value=750, help="Adjust the width of the generated tables.")
        debug_attempts = 3
        share_data = st.session_state['share_chat_history_button']
        color_coded_data = st.session_state['display_colored_state']

    with tab_selected[2]:
        st.header("Feedback")
        feedback = st.text_area("Please provide your feedback here:", height=150)
        if st.button(":material/check: Submit", help="Click to submit feedback"):
            if feedback:            
                save_feedback_locally(feedback, local_file_path_feedback)
                upload_feedback_to_s3(
                    s3_object=s3,
                    file_path=local_file_path_feedback,
                    bucket=bucket_name, 
                    s3_key=s3_file_path_feedback
                )
            st.toast("Thank you for submitting feedback", icon=":material/thumb_up:")

    status_placeholder = st.empty()
    error_occurred = False

    fiscal_month, fiscal_quarter, fiscal_year = get_current_fiscal_dates()

    if st.session_state['task_selected_state'] == "Compare changes between data sources":

        with st.expander("Click to view the data", expanded=True):

            col1, col2 = st.columns(2)
            files_list = get_local_files()

            with col1:
                files_list_new = [file.split("/")[-1] for file in files_list]
                first_file = st.selectbox("Select the file to analyze:", files_list_new)
                updated_files_list = files_list.copy()
                dfs = loading_local_csv(file=first_file)
                dfs = divide_data(data=dfs, divider=number)


            with col2:
                updated_files_list_new = [file.split("/")[-1] for file in updated_files_list]
                updated_files_list_new.remove(first_file)
                second_file = st.selectbox("Select the second file to analyze:", updated_files_list_new)
                dfs2 = loading_local_csv(file=second_file)
                dfs2 = divide_data(data=dfs2, divider=number)

            drop_columns = [
                'Part #', 
                'Customer #', 
                'End Customer', 
                'Sale Ord #', 
                'Line #', 
                'Sch Ln#', 
                'PO #', 
                'Del #', 
                'Del Date', 
                'Product End Customer',
                'Product End Customer Name',
                'Rejection',
                'Planning Group',
                'Higher Level Item'
            ]

            for column_selected in dfs.columns:
                if column_selected in drop_columns:
                    dfs.drop([column_selected], axis=1, inplace=True)
                    dfs2.drop([column_selected], axis=1, inplace=True)

            dfs = dfs[dfs['Demand Type'] != 'RSF']
            dfs2 = dfs2[dfs2['Demand Type'] != 'RSF']

            col1, col2 = st.columns(2)

            product_category_choices = [
                'Business Unit',
                'Family',
                'Planning Group',
                'SKU'
            ]
            
            dfs_copy = dfs.copy()
            dfs2_copy = dfs2.copy()

            with col1:
                product_category_selected = st.selectbox("Select the Product Category:", product_category_choices)
            with col2:
                if product_category_selected == "Business Unit":
                    business_unit_selected = st.selectbox("Select the Business Unit:", dfs_copy['Business Unit'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['Business Unit'] == business_unit_selected].copy()
                    dfs2_copy = dfs2_copy[dfs2_copy['Business Unit'] == business_unit_selected].copy()
                elif product_category_selected == "Family":
                    product_family_selected = st.selectbox("Select the Product Family:", dfs_copy['Family'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['Family'] == product_family_selected].copy()
                    dfs2_copy = dfs2_copy[dfs2_copy['Family'] == product_family_selected].copy()
                elif product_category_selected == "Planning Group":
                    planning_group_selected = st.selectbox("Select the Planning Group:", dfs_copy['Planning Group'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['Planning Group'] == planning_group_selected].copy()
                    dfs2_copy = dfs2_copy[dfs2_copy['Planning Group'] == planning_group_selected].copy()
                else:
                    sku_selected = st.selectbox("Select the SKU:", dfs_copy['SKU'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['SKU'] == sku_selected].copy()
                    dfs2_copy = dfs2_copy[dfs2_copy['SKU'] == sku_selected].copy()

            dfs_copy.reset_index(drop=True, inplace=True)
            dfs2_copy.reset_index(drop=True, inplace=True)
            empty_columns = dfs_copy.columns[dfs_copy.isna().all()]
            empty_columns2 = dfs2_copy.columns[dfs2_copy.isna().all()]
            dfs_copy.drop(columns=empty_columns, inplace=True)
            dfs2_copy.drop(columns=empty_columns2, inplace=True)
            dfs = dfs_copy.copy()
            dfs2 = dfs2_copy.copy()

            st.sidebar.divider()
            if st.sidebar.button(":material/menu: Anaplan View"):
                df_time_profile = pd.read_csv(os.path.abspath("data/Time Profile for BOT.csv"))
                df_time_profile['Day'] = pd.to_datetime(df_time_profile['Day'])
                df_time_profile['Weekly Dates'] = df_time_profile['Day'].apply(lambda x: x + pd.Timedelta(days=(6 - x.weekday())) if x.weekday() <= 6 else None)
                df_time_profile['Weekly Dates'] = df_time_profile['Weekly Dates'].apply(lambda x: f"W/e {x.strftime('%d %b %y')}" if x else None)
                current_date = date.today()
                current_date_pandas = pd.to_datetime(current_date)
                current_fiscal_quarter = df_time_profile[df_time_profile['Day'] == current_date_pandas]['Fiscal Quarter'].values[0]
                weekly_dates = list(df_time_profile[df_time_profile['Fiscal Quarter'] == current_fiscal_quarter]['Weekly Dates'].unique()) + [current_fiscal_quarter]
                df_result = index_time_profile(
                    df_time_profile=df_time_profile,
                    current_fiscal_quarter=current_fiscal_quarter
                )
                dfs_copy = dfs.copy()
                dfs2_copy = dfs2.copy()
                dfs_copy.loc[dfs_copy['Demand Type'] == 'PULLED', 'Demand Type'] = 'SHIPPED'
                dfs_copy.loc[dfs_copy['Demand Type'] == 'ZSCH', 'Demand Type'] = 'BOOKED'
                dfs2_copy.loc[dfs2_copy['Demand Type'] == 'PULLED', 'Demand Type'] = 'SHIPPED'
                dfs2_copy.loc[dfs2_copy['Demand Type'] == 'ZSCH', 'Demand Type'] = 'BOOKED'

                anaplan_dataframe = convert_to_anaplan(
                    df_date_converted=dfs_copy,
                    df_result=df_result,
                    current_fiscal_quarter=current_fiscal_quarter
                )

                anaplan_dataframe2 = convert_to_anaplan(
                    df_date_converted=dfs2_copy,
                    df_result=df_result,
                    current_fiscal_quarter=current_fiscal_quarter
                )
                anaplan_view_compare_changes(anaplan_dataframe, anaplan_dataframe2)

            st.write(f"First file selected: {first_file}")
            st.dataframe(dfs, height=200)
            st.write(f"Second file selected: {second_file}")
            st.dataframe(dfs2, height=200)

            # Get and display categorical columns
            categorical_columns_dict = get_categorical_columns(dfs)

            dfs_cat, dfs2_cat, intersect_cat = compare_categorical_columns(dfs, dfs2)

            # Get and display numerical columns
            numerical_columns_dict = get_numerical_columns(dfs)

            datetime_columns = get_datetime_columns(dfs)

            for datetime_column in datetime_columns:
                dfs[datetime_column] = pd.to_datetime(dfs[datetime_column])

            # Get and display categorical columns
            categorical_columns_dict2 = get_categorical_columns(dfs2)

            # Get and display numerical columns
            numerical_columns_dict2 = get_numerical_columns(dfs2)

            datetime_columns2 = get_datetime_columns(dfs2)

            for datetime_column in datetime_columns2:
                dfs2[datetime_column] = pd.to_datetime(dfs2[datetime_column])

            system_message_file_uploaded = system_message_compare_changes.format(
                first_file=first_file,
                dfs_cat=dfs_cat,
                numerical_columns_dict=numerical_columns_dict,
                datetime_columns=datetime_columns,
                second_file=second_file,
                dfs2_cat=dfs2_cat,
                numerical_columns_dict2=numerical_columns_dict2,
                datetime_columns2=datetime_columns2,
                intersect_cat=intersect_cat,
                categorical_columns_dict2=categorical_columns_dict2,
                fiscal_month=fiscal_month, 
                fiscal_quarter=fiscal_quarter,
                fiscal_year=fiscal_year
            )

        if "messages" not in st.session_state:

            st.session_state.messages = ["placeholder"]

        st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}

        with tab_selected[1]:

            if st.button(":material/chat: Click for new chat", help="Click for new chat"):
                st.session_state.flag_new_chat = 1
                now = datetime.now()
                st.session_state.new_formatted_date_time = now.strftime("%B %d, %Y %I:%M:%S %p")
                st.session_state.conversations[f"{st.session_state.new_formatted_date_time}"] = []
                st.session_state.conversation_selected = st.session_state.new_formatted_date_time
                st.session_state.messages = []
                st.session_state.messages_display = []
                st.session_state.debugger_messages = []
                st.session_state.messages = ["placeholder"]
                st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}
                st.session_state.button_clicked_flag = False
            st.markdown('<hr style="border:0.01px solid #cccccc">', unsafe_allow_html=True)
            for selected_conversation in list(st.session_state.conversations.keys()):
                if st.button(selected_conversation):
                    st.session_state.conversation_selected = selected_conversation
                    st.session_state.button_clicked_flag = True
                    st.session_state.messages_display = st.session_state.conversations[selected_conversation]

        if st.session_state.flag_new_chat == 0:
            st.session_state.conversation_selected = st.session_state.old_formatted_date_time
            st.session_state.conversations[f'{st.session_state.conversation_selected}'] = st.session_state.messages_display

        if st.session_state.flag_new_chat == 1:
            if st.session_state.button_clicked_flag == False:
                st.session_state.conversation_selected == st.session_state.new_formatted_date_time
                st.session_state.conversations[st.session_state.conversation_selected] == st.session_state.messages_display

        for message in st.session_state.conversations[st.session_state.conversation_selected]:
            if message['role'] == 'user':
                with st.chat_message(message['role'], avatar="🔍"):
                    st.markdown(message['content'])
            if message['role'] == 'assistant':
                with st.status("📟 *Generating the code*..."):
                    with st.chat_message(message['role'], avatar="🤖"):
                        st.markdown(message['content'])
            if message['role'] == 'plot':
                st.plotly_chart(message['figure'])
            if message['role'] == 'adhoc':
                st.write(message['message from adhoc query'])
            if message['role'] == 'show_diff_data':
                st.dataframe(message['dataframe'], width=column_width)

        if generate_prompt_suggestions:
            recommender_selected = Recommender(
                llm_model=st.session_state['recommender'],
                private_nvcf_endpoints=private_nvcf_endpoints
            )

            st.sidebar.divider()
            total_suggestions = st.sidebar.number_input("Total Prompt Suggestions:", value=5)
            recommender_message = {"role": "user", "content": f"Based on the context, give a total of {total_suggestions} suggestions for prompts which demand planner can likely ask. Here is the context so far: {st.session_state.messages}. Each prompt suggestion MUST end with \n"}
            llm_suggestions = recommender_selected.chat(
                messages=recommender_message,
            )

            llm_suggestions = llm_suggestions.choices[0].message.content
            llm_suggestions_list = [llm_suggestion for llm_suggestion in llm_suggestions.split('\n\n') if llm_suggestion.strip()]

            for llm_suggestion in llm_suggestions_list:
                if st.sidebar.button(llm_suggestion):

                    additional_message = f"""
                    INSTRUCTIONS 
                    - If it is a general purpose query, let the user only know your purpose briefly.
                    - Use only one function to answer all the questions asked by the user and nothing else. ALWAYS ensure to give only llm_response(df, df2) function.
                    - Import the necessary libraries which are needed for the task for running this function. If libraries are not needed, you need not import them. 
                    - Try to give as accurate and executable code as possible without syntax errors.

                    a) llm_response(df, df2) - Use this function to create a plot, insights from data and tables.
                    The output should be of the list format [].
                    The elements in the list can 'fig' which is a figure, 'insights' which is a string and 'table' which is a dataframe table.
                    You can decide whether to use one element, two elements or all three elements in the list or more.
                    You can also use more elements but ensure that they are similar in datatypes such as 'fig', 'insights' and 'table' respectively. 
                    
                    Be sure to use only NVIDIA colors #76B900 when plotting.

                    # ALWAYS use current quarter. The following is the current quarter: {fiscal_quarter}

                    # Here is the current month if needed: {fiscal_month}

                    # Here is the current fiscal year if needed: {fiscal_year}
                    """

                    # which is an ExcelFile format along with 'sheet_name'. The 'sheet_name' should be a keyword argument.
                    enhanced_prompt = llm_suggestion + additional_message
                    st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
                    st.session_state.messages_display.append({'role': 'user', 'content': llm_suggestion})
                    with st.chat_message("user", avatar="🔍"):
                        st.markdown(llm_suggestion)                    

                    status_placeholder = st.empty()
                    error_occurred = False
                    metadata = {
                        'categorical_columns_dict': categorical_columns_dict,
                        'numerical_columns_dict': numerical_columns_dict,
                        'datetime_columns': datetime_columns,
                    }

                    # st.write(st.session_state.messages)

                    with status_placeholder.status("📟 *Generating the code*..."):
                        with st.chat_message("assistant", avatar="🤖"):
                            if not private_nvcf_endpoints:
                                stream = client.chat.completions.create(
                                    model=st.session_state["llm_model"],
                                    messages=[
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.messages
                                    ],
                                    stream=True,
                                    temperature=0.0,
                                    max_tokens=4096
                                )
                
                                result = stream_responses(input_stream=stream)   
                            else:
                                response_list = []
                                botmsg = st.empty()
                                for chunk in client.stream(
                                    [
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.messages
                                    ]
                                ):
                                    response_list.append(chunk.content)
                                    result = "".join(response_list)
                                    botmsg.write(result + "▌")
                                if result:
                                    botmsg.write(result) 


                    st.session_state.messages.append({"role": "assistant", "content": result})
                    st.session_state.messages_display.append({'role': 'assistant', 'content': result})
                    code_generated = extract_code(result)

                    with open("llm_generated_code.py", 'r') as file:
                        exec_namespace = {}
                        file_read = file.read()
                        exec(file_read, exec_namespace)

                        if 'show_diff_data' in file_read:
                            try:
                                df_generated = exec_namespace['show_diff_data'](dfs, dfs2)
                                if color_coded_data == "Yes":
                                    styled_df = df_generated.style.applymap(color_code)
                                    st.dataframe(styled_df, width=column_width)
                                    st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': styled_df})   
                                else:
                                    st.dataframe(df_generated, width=column_width)
                                    st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': df_generated})   
                            except Exception as e:
                                error_occurred = True
                                debugger_selected = Debugger(
                                    llm_model=st.session_state['debugger'],
                                    kwargs=metadata,
                                    private_nvcf_endpoints=private_nvcf_endpoints
                                )
                                with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                                    while error_occurred: 
                                        error_count = 0
                                        error_message = traceback.format_exc()
                                        correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                                        stream = debugger_selected.chat(
                                            messages=correction_message,
                                        )
                                        result = stream_responses(input_stream=stream) 
                                        st.session_state.messages.append(correction_message)
                                        result_message = {"role": "assistant", "content": result}
                                        st.session_state.messages.append(result_message)
                                        code_generated = extract_code(result)

                                        with open("llm_generated_code.py", 'r') as file:
                                            exec_namespace = {}
                                            file_read = file.read()
                                        try:
                                            exec(file_read, exec_namespace)
                                            df_generated = exec_namespace['show_diff_data'](dfs, dfs2)
                                            error_occurred = False
                                        except:
                                            error_count = error_count + 1
                                            if error_count > debug_attempts:
                                                error_occurred = False

                                if color_coded_data == "Yes":
                                    styled_df = df_generated.style.applymap(color_code)
                                    st.dataframe(styled_df, width=column_width)
                                    st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': styled_df})   
                                else:
                                    st.dataframe(df_generated, width=column_width)
                                    st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': df_generated})   

                        if 'data_difference' in file_read:
                            try:
                                exec(file_read, exec_namespace)
                                message_returned = exec_namespace['data_difference'](dfs, dfs2)
                                # st.write(message_returned)

                                if not private_nvcf_endpoints:

                                    client = OpenAI(
                                        base_url=base_url,
                                        api_key=api_key
                                    )

                                    stream = client.chat.completions.create(
                                        model=st.session_state["llm_model"],
                                        messages=[
                                            {'role': 'user', 'content': message_returned + ". Rewrite this so that it is easier to understand and only the statement should appear and not additional information. When giving answers, try to make it legible and easy for user to see the output."}
                                        ],
                                        stream=True,
                                        temperature=0.0,
                                        max_tokens=4096
                                    )

                                    result = stream_responses(input_stream=stream) 
                                else:
                                    response_list = []
                                    botmsg = st.empty()
                                    for chunk in client.stream(
                                        [
                                            {"role": m["role"], "content": m["content"]}
                                            for m in st.session_state.messages
                                        ]
                                    ):
                                        response_list.append(chunk.content)
                                        result = "".join(response_list)
                                        botmsg.write(result + "▌")
                                    if result:
                                        botmsg.write(result) 

                                st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})
                            except Exception as e:
                                error_occurred = True
                                debugger_selected = Debugger(
                                    llm_model=st.session_state['debugger'],
                                    kwargs=metadata,
                                    private_nvcf_endpoints=private_nvcf_endpoints
                                )
                                with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                                    while error_occurred: 
                                        error_count = 0
                                        error_message = traceback.format_exc()
                                        correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                                        stream = debugger_selected.chat(
                                            messages=correction_message,
                                        )
                                        result = stream_responses(input_stream=stream) 
                                        st.session_state.messages.append(correction_message)
                                        result_message = {"role": "assistant", "content": result}
                                        st.session_state.messages.append(result_message)
                                        code_generated = extract_code(result)

                                        with open("llm_generated_code.py", 'r') as file:
                                            exec_namespace = {}
                                            file_read = file.read()
                                        try:
                                            exec(file_read, exec_namespace)
                                            message_returned = exec_namespace['data_difference'](dfs, dfs2)
                                            error_occurred = False
                                        except:
                                            error_count = error_count + 1
                                            if error_count > debug_attempts:
                                                error_occurred = False

                                if not private_nvcf_endpoints:
                                    stream = client.chat.completions.create(
                                        model=st.session_state["llm_model"],
                                        messages=[
                                            {'role': 'user', 'content': message_returned + ". Rewrite this so that it is easier to understand and only the statement should appear and not additional information. Note that these are the orders for NVIDIA from various customers."}
                                        ],
                                        stream=True,
                                        temperature=0.0,
                                        max_tokens=4096
                                    )

                                    result = stream_responses(input_stream=stream) 
                                else:
                                    response_list = []
                                    botmsg = st.empty()
                                    for chunk in client.stream(
                                        [
                                            {"role": m["role"], "content": m["content"]}
                                            for m in st.session_state.messages
                                        ]
                                    ):
                                        response_list.append(chunk.content)
                                        result = "".join(response_list)
                                        botmsg.write(result + "▌")
                                    if result:
                                        botmsg.write(result) 
                                st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})

                        if 'plot_changes' in file_read:
                            try:
                                exec(file_read, exec_namespace)
                                fig_returned = exec_namespace['plot_changes'](dfs, dfs2)
                                if not fig_returned.data:
                                    st.warning("The figure is empty")
                                else:
                                    st.plotly_chart(fig_returned)
                                    st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned})   
                            except Exception as e:
                                error_occurred = True
                                debugger_selected = Debugger(
                                    llm_model=st.session_state['debugger'],
                                    kwargs=metadata,
                                    private_nvcf_endpoints=private_nvcf_endpoints
                                )
                                with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                                    while error_occurred: 
                                        error_count = 0
                                        error_message = traceback.format_exc()
                                        correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                                        stream = debugger_selected.chat(
                                            messages=correction_message,
                                        )
                                        result = stream_responses(input_stream=stream) 
                                        st.session_state.messages.append(correction_message)
                                        result_message = {"role": "assistant", "content": result}
                                        st.session_state.messages.append(result_message)
                                        code_generated = extract_code(result)

                                        with open("llm_generated_code.py", 'r') as file:
                                            exec_namespace = {}
                                            file_read = file.read()
                                        try:
                                            exec(file_read, exec_namespace)
                                            fig_returned = exec_namespace['plot_changes'](dfs, dfs2)
                                            error_occurred = False
                                        except:
                                            error_count = error_count + 1
                                            if error_count > debug_attempts:
                                                error_occurred = False

                                st.plotly_chart(fig_returned)
                                st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned}) 

                    if share_data == "Yes":
                        save_chat_history_locally(
                            data_list=st.session_state.messages, 
                            file_path='chat_history.xlsx'
                        )
                        upload_chat_history_to_s3(
                            s3_object=s3, 
                            file_path=local_file_path_chat_history, 
                            bucket=bucket_name, 
                            s3_key=s3_file_path_chat_history
                        )

                    # Clearing the LLM generated code 
                    with open("llm_generated_code.py", "w") as file:
                        pass


        if prompt := st.chat_input("Write your lines here..."):

            additional_message = f"""
            INSTRUCTIONS 
            - If it is a general purpose query, let the user only know your purpose briefly.
            - Use only one function to answer all the questions asked by the user and nothing else. ALWAYS ensure to give only llm_response(df, df2) function.
            - Import the necessary libraries which are needed for the task for running this function. If libraries are not needed, you need not import them. 
            - Try to give as accurate and executable code as possible without syntax errors.

            a) llm_response(df, df2) - Use this function to create a plot, insights from data and tables.
            The output should be of the list format [].
            The elements in the list can 'fig' which is a figure, 'insights' which is a string and 'table' which is a dataframe table.
            You can decide whether to use one element, two elements or all three elements in the list or more.
            You can also use more elements but ensure that they are similar in datatypes such as 'fig', 'insights' and 'table' respectively. 
            
            Be sure to use only NVIDIA colors #76B900 when plotting.

            # ALWAYS use current quarter. The following is the current quarter: {fiscal_quarter}

            # Here is the current month if needed: {fiscal_month}

            # Here is the current fiscal year if needed: {fiscal_year}
            """

            # which is an ExcelFile format along with 'sheet_name'. The 'sheet_name' should be a keyword argument.
            enhanced_prompt = prompt + additional_message
            st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
            st.session_state.messages_display.append({'role': 'user', 'content': prompt})
            with st.chat_message("user", avatar="🔍"):
                st.markdown(prompt)

            # st.write(st.session_state.messages)

            
            # st.write([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
            

            status_placeholder = st.empty()
            error_occurred = False
            metadata = {
                'categorical_columns_dict': categorical_columns_dict,
                'numerical_columns_dict': numerical_columns_dict,
                'datetime_columns': datetime_columns,
            }

            # st.write(st.session_state.messages)

            with status_placeholder.status("📟 *Generating the code*..."):
                with st.chat_message("assistant", avatar="🤖"):
                    if not private_nvcf_endpoints:
                        stream = client.chat.completions.create(
                            model=st.session_state["llm_model"],
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                            temperature=0.0,
                            max_tokens=4096
                        )
        
                        result = stream_responses(input_stream=stream)   
                    else:
                        response_list = []
                        botmsg = st.empty()
                        for chunk in client.stream(
                            [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ]
                        ):
                            response_list.append(chunk.content)
                            result = "".join(response_list)
                            botmsg.write(result + "▌")
                        if result:
                            botmsg.write(result) 


            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.messages_display.append({'role': 'assistant', 'content': result})
            code_generated = extract_code(result)

            with open("llm_generated_code.py", 'r') as file:
                exec_namespace = {}
                file_read = file.read()
                exec(file_read, exec_namespace)

                if 'show_diff_data' in file_read:
                    try:
                        df_generated = exec_namespace['show_diff_data'](dfs, dfs2)
                        if color_coded_data == "Yes":
                            styled_df = df_generated.style.applymap(color_code)
                            st.dataframe(styled_df, width=column_width)
                            st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': styled_df})   
                        else:
                            st.dataframe(df_generated, width=column_width)
                            st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': df_generated})   
                    except Exception as e:
                        error_occurred = True
                        debugger_selected = Debugger(
                            llm_model=st.session_state['debugger'],
                            kwargs=metadata,
                            private_nvcf_endpoints=private_nvcf_endpoints
                        )
                        with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                            while error_occurred: 
                                error_count = 0
                                error_message = traceback.format_exc()
                                correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                                stream = debugger_selected.chat(
                                    messages=correction_message,
                                )
                                result = stream_responses(input_stream=stream) 
                                st.session_state.messages.append(correction_message)
                                result_message = {"role": "assistant", "content": result}
                                st.session_state.messages.append(result_message)
                                code_generated = extract_code(result)

                                with open("llm_generated_code.py", 'r') as file:
                                    exec_namespace = {}
                                    file_read = file.read()
                                try:
                                    exec(file_read, exec_namespace)
                                    df_generated = exec_namespace['show_diff_data'](dfs, dfs2)
                                    error_occurred = False
                                except:
                                    error_count = error_count + 1
                                    if error_count > debug_attempts:
                                        error_occurred = False

                        if color_coded_data == "Yes":
                            styled_df = df_generated.style.applymap(color_code)
                            st.dataframe(styled_df, width=column_width)
                            st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': styled_df})   
                        else:
                            st.dataframe(df_generated, width=column_width)
                            st.session_state.messages_display.append({'role': 'show_diff_data', 'dataframe': df_generated})   

                if 'data_difference' in file_read:
                    try:
                        exec(file_read, exec_namespace)
                        message_returned = exec_namespace['data_difference'](dfs, dfs2)
                        # st.write(message_returned)

                        if not private_nvcf_endpoints:

                            client = OpenAI(
                                base_url=base_url,
                                api_key=api_key
                            )

                            stream = client.chat.completions.create(
                                model=st.session_state["llm_model"],
                                messages=[
                                    {'role': 'user', 'content': message_returned + ". Rewrite this so that it is easier to understand and only the statement should appear and not additional information. When giving answers, try to make it legible and easy for user to see the output."}
                                ],
                                stream=True,
                                temperature=0.0,
                                max_tokens=4096
                            )

                            result = stream_responses(input_stream=stream) 
                        else:
                            response_list = []
                            botmsg = st.empty()
                            for chunk in client.stream(
                                [
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                ]
                            ):
                                response_list.append(chunk.content)
                                result = "".join(response_list)
                                botmsg.write(result + "▌")
                            if result:
                                botmsg.write(result) 

                        st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})
                    except Exception as e:
                        error_occurred = True
                        debugger_selected = Debugger(
                            llm_model=st.session_state['debugger'],
                            kwargs=metadata,
                            private_nvcf_endpoints=private_nvcf_endpoints
                        )
                        with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                            while error_occurred: 
                                error_count = 0
                                error_message = traceback.format_exc()
                                correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                                stream = debugger_selected.chat(
                                    messages=correction_message,
                                )
                                result = stream_responses(input_stream=stream) 
                                st.session_state.messages.append(correction_message)
                                result_message = {"role": "assistant", "content": result}
                                st.session_state.messages.append(result_message)
                                code_generated = extract_code(result)

                                with open("llm_generated_code.py", 'r') as file:
                                    exec_namespace = {}
                                    file_read = file.read()
                                try:
                                    exec(file_read, exec_namespace)
                                    message_returned = exec_namespace['data_difference'](dfs, dfs2)
                                    error_occurred = False
                                except:
                                    error_count = error_count + 1
                                    if error_count > debug_attempts:
                                        error_occurred = False

                        if not private_nvcf_endpoints:
                            stream = client.chat.completions.create(
                                model=st.session_state["llm_model"],
                                messages=[
                                    {'role': 'user', 'content': message_returned + ". Rewrite this so that it is easier to understand and only the statement should appear and not additional information. Note that these are the orders for NVIDIA from various customers."}
                                ],
                                stream=True,
                                temperature=0.0,
                                max_tokens=4096
                            )

                            result = stream_responses(input_stream=stream) 
                        else:
                            response_list = []
                            botmsg = st.empty()
                            for chunk in client.stream(
                                [
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                ]
                            ):
                                response_list.append(chunk.content)
                                result = "".join(response_list)
                                botmsg.write(result + "▌")
                            if result:
                                botmsg.write(result) 
                        st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})

                if 'plot_changes' in file_read:
                    try:
                        exec(file_read, exec_namespace)
                        fig_returned = exec_namespace['plot_changes'](dfs, dfs2)
                        if not fig_returned.data:
                            st.warning("The figure is empty")
                        else:
                            st.plotly_chart(fig_returned)
                            st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned})   
                    except Exception as e:
                        error_occurred = True
                        debugger_selected = Debugger(
                            llm_model=st.session_state['debugger'],
                            kwargs=metadata,
                            private_nvcf_endpoints=private_nvcf_endpoints
                        )
                        with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                            while error_occurred: 
                                error_count = 0
                                error_message = traceback.format_exc()
                                correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                                stream = debugger_selected.chat(
                                    messages=correction_message,
                                )
                                result = stream_responses(input_stream=stream) 
                                st.session_state.messages.append(correction_message)
                                result_message = {"role": "assistant", "content": result}
                                st.session_state.messages.append(result_message)
                                code_generated = extract_code(result)

                                with open("llm_generated_code.py", 'r') as file:
                                    exec_namespace = {}
                                    file_read = file.read()
                                try:
                                    exec(file_read, exec_namespace)
                                    fig_returned = exec_namespace['plot_changes'](dfs, dfs2)
                                    error_occurred = False
                                except:
                                    error_count = error_count + 1
                                    if error_count > debug_attempts:
                                        error_occurred = False

                        st.plotly_chart(fig_returned)
                        st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned}) 

            if share_data == "Yes":
                save_chat_history_locally(
                    data_list=st.session_state.messages, 
                    file_path='chat_history.xlsx'
                )
                upload_chat_history_to_s3(
                    s3_object=s3, 
                    file_path=local_file_path_chat_history, 
                    bucket=bucket_name, 
                    s3_key=s3_file_path_chat_history
                )

            # Clearing the LLM generated code 
            with open("llm_generated_code.py", "w") as file:
                pass

    else:
        fiscal_month, fiscal_quarter, fiscal_year = get_current_fiscal_dates()
        # files_list = get_local_files()
        files_list = get_aws_files()
        with st.expander("Click to view the data", expanded=True):
            # files_list_new = [file.split("/")[-1] for file in files_list]
            # first_file = st.selectbox("Select the file to analyze:", files_list_new)
            # dfs = loading_local_csv(file=first_file)
            # dfs = divide_data(data=dfs, divider=number)


            first_file = st.selectbox("Select the file to analyze:", files_list)
            dfs = loading_csv(file=first_file)
            dfs = divide_data(data=dfs, divider=number)

            drop_columns = [
                'Part #', 
                'Adaptor',
                'Customer #', 
                'End Customer', 
                'Sale Ord #', 
                'Line #', 
                'Sch Ln#', 
                'PO #', 
                'Del #', 
                'Del Date', 
                'Product End Customer',
                'Product End Customer Name',
                'Rejection',
                'Higher Level Item',
                'Unit Price',
                'Amount'
            ]
            for column_selected in dfs.columns:
                if column_selected in drop_columns:
                    dfs.drop([column_selected], axis=1, inplace=True)
            dfs = dfs[dfs['Demand Type'] != 'RSF']
            col1, col2 = st.columns(2)
            product_category_choices = [
                'Business Unit',
                'Family',
                'Planning Group',
                'SKU'
            ]
            dfs_copy = dfs.copy()
            with col1:
                product_category_selected = st.selectbox("Select the Product Category:", product_category_choices)
            with col2:
                if product_category_selected == "Business Unit":
                    business_unit_selected = st.selectbox("Select the Business Unit:", dfs_copy['Business Unit'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['Business Unit'] == business_unit_selected].copy()
                elif product_category_selected == "Family":
                    product_family_selected = st.selectbox("Select the Product Family:", dfs_copy['Family'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['Family'] == product_family_selected].copy()
                elif product_category_selected == "Planning Group":
                    planning_group_selected = st.selectbox("Select the Planning Group:", dfs_copy['Planning Group'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['Planning Group'] == planning_group_selected].copy()
                else:
                    sku_selected = st.selectbox("Select the SKU:", dfs_copy['SKU'].unique(), index=1)
                    dfs_copy = dfs_copy[dfs_copy['SKU'] == sku_selected].copy()
            dfs_copy.reset_index(drop=True, inplace=True)
            empty_columns = dfs_copy.columns[dfs_copy.isna().all()]
            dfs_copy.drop(columns=empty_columns, inplace=True)
            dfs = dfs_copy.copy()
            if st.sidebar.button(":material/menu: Anaplan View"):
                df_time_profile = pd.read_csv(os.path.abspath("data/Time Profile for BOT.csv"))
                df_time_profile['Day'] = pd.to_datetime(df_time_profile['Day'])
                df_time_profile['Weekly Dates'] = df_time_profile['Day'].apply(lambda x: x + pd.Timedelta(days=(6 - x.weekday())) if x.weekday() <= 6 else None)
                df_time_profile['Weekly Dates'] = df_time_profile['Weekly Dates'].apply(lambda x: f"W/e {x.strftime('%d %b %y')}" if x else None)
                current_date = date.today()
                current_date_pandas = pd.to_datetime(current_date)
                current_fiscal_quarter = df_time_profile[df_time_profile['Day'] == current_date_pandas]['Fiscal Quarter'].values[0]
                df_result = index_time_profile(
                    df_time_profile=df_time_profile,
                    current_fiscal_quarter=current_fiscal_quarter
                )
                dfs_copy_anaplan = dfs.copy()
                dfs_copy_anaplan.loc[dfs_copy_anaplan['Demand Type'] == 'PULLED', 'Demand Type'] = 'SHIPPED'
                dfs_copy_anaplan.loc[dfs_copy_anaplan['Demand Type'] == 'ZSCH', 'Demand Type'] = 'BOOKED'
                anaplan_dataframe = convert_to_anaplan(
                    df_date_converted=dfs_copy_anaplan,
                    df_result=df_result,
                    current_fiscal_quarter=current_fiscal_quarter
                )
                anaplan_view(anaplan_dataframe)
            
            st.dataframe(dfs)

            categorical_columns_dict = get_categorical_columns(dfs)
            numerical_columns_dict = get_numerical_columns(dfs)
            datetime_columns = get_datetime_columns(dfs)

            for datetime_column in datetime_columns:
                dfs[datetime_column] = pd.to_datetime(dfs[datetime_column])

            system_message_file_uploaded = system_message_analyze.format(
                first_file=first_file,
                categorical_columns_dict=categorical_columns_dict,
                numerical_columns_dict=numerical_columns_dict,
                datetime_columns=datetime_columns,
                fiscal_month=fiscal_month, 
                fiscal_quarter=fiscal_quarter,
                fiscal_year=fiscal_year
            )

        if "messages" not in st.session_state:

            st.session_state.messages = ["placeholder"]

        st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}

        with tab_selected[1]:

            if st.button(":material/chat: Click for new chat", help="Click for new chat"):
                st.session_state.flag_new_chat = 1
                now = datetime.now()
                st.session_state.new_formatted_date_time = now.strftime("%B %d, %Y %I:%M:%S %p")
                st.session_state.conversations[f"{st.session_state.new_formatted_date_time}"] = []
                st.session_state.conversation_selected = st.session_state.new_formatted_date_time
                st.session_state.messages = []
                st.session_state.messages_display = []
                st.session_state.debugger_messages = []
                st.session_state.messages = ["placeholder"]
                st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}
                st.session_state.button_clicked_flag = False
            st.markdown('<hr style="border:0.01px solid #cccccc">', unsafe_allow_html=True)
            for selected_conversation in list(st.session_state.conversations.keys()):
                if st.button(selected_conversation):
                    st.session_state.conversation_selected = selected_conversation
                    st.session_state.button_clicked_flag = True
                    st.session_state.messages_display = st.session_state.conversations[selected_conversation]

            st.write(f"**Conversation selected:** *{st.session_state.conversation_selected}*")

        if st.session_state.flag_new_chat == 0:
            st.session_state.conversation_selected = st.session_state.old_formatted_date_time
            st.session_state.conversations[f'{st.session_state.conversation_selected}'] = st.session_state.messages_display

        if st.session_state.flag_new_chat == 1:
            if st.session_state.button_clicked_flag == False:
                st.session_state.conversation_selected = st.session_state.new_formatted_date_time
                st.session_state.conversations[st.session_state.conversation_selected] = st.session_state.messages_display

        for message in st.session_state.conversations[st.session_state.conversation_selected]:
            if message['role'] == 'user':
                with st.chat_message(message['role'], avatar="🔍"):
                    st.markdown(message['content'])
            if message['role'] == 'assistant':
                with st.status("📟 *Generating the code*..."):
                    with st.chat_message(message['role'], avatar="🤖"):
                        st.markdown(message['content'])
            if message['role'] == 'plot':
                st.plotly_chart(message['figure'], key=str(uuid.uuid4()))
            if message['role'] == 'adhoc':
                st.write(message['message from adhoc query'])
            if message['role'] == 'show_data' or message['role'] == 'show_diff_data':
                st.dataframe(message['dataframe'], width=column_width)

        if generate_prompt_suggestions:
            recommender_selected = Recommender(
                llm_model=st.session_state['recommender'],
                private_nvcf_endpoints=private_nvcf_endpoints
            )

            st.sidebar.divider()
            total_suggestions = st.sidebar.number_input("Total Prompt Suggestions:", value=5)
            recommender_message = {"role": "user", "content": f"Based on the context, give a total of {total_suggestions} suggestions for prompts which demand planner can likely ask. Here is the context so far: {st.session_state.messages}. Each prompt suggestion MUST end with \n"}
            llm_suggestions = recommender_selected.chat(
                messages=recommender_message,
            )

            llm_suggestions = llm_suggestions.choices[0].message.content
            llm_suggestions_list = [llm_suggestion for llm_suggestion in llm_suggestions.split('\n\n') if llm_suggestion.strip()]

            for llm_suggestion in llm_suggestions_list:
                if st.sidebar.button(llm_suggestion):

                    additional_message = f"""
                    INSTRUCTIONS 
                    - If it is a general purpose query, let the user only know your purpose briefly.
                    - Use only one function to answer all the questions asked by the user and nothing else. ALWAYS ensure to give only llm_response(df) function.
                    - Import the necessary libraries which are needed for the task for running this function. If libraries are not needed, you need not import them. 
                    - Try to give as accurate and executable code as possible without syntax errors.

                    a) llm_response(df) - Use this function to create a plot, insights from data and tables.
                    The output should be of the list format [].
                    The elements in the list can 'fig' which is a figure, 'insights' which is a string and 'table' which is a dataframe table.
                    You can decide whether to use one element, two elements or all three elements in the list or more.
                    You can also use more elements but ensure that they are similar in datatypes such as 'fig', 'insights' and 'table' respectively. 
                    
                    Be sure to use only NVIDIA colors #76B900 when plotting.

                    # ALWAYS use current quarter. The following is the current quarter: {fiscal_quarter}

                    # Here is the current month if needed: {fiscal_month}

                    # Here is the current fiscal year if needed: {fiscal_year}
                    """

                    # which is an ExcelFile format along with 'sheet_name'. The 'sheet_name' should be a keyword argument.
                    enhanced_prompt = llm_suggestion + additional_message
                    st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
                    st.session_state.messages_display.append({'role': 'user', 'content': llm_suggestion})
                    with st.chat_message("user", avatar="🔍"):
                        st.markdown(llm_suggestion)

                    status_placeholder = st.empty()
                    error_occurred = False
                    metadata = {
                        'categorical_columns_dict': categorical_columns_dict,
                        'numerical_columns_dict': numerical_columns_dict,
                        'datetime_columns': datetime_columns,
                    }

                    # st.write(st.session_state.messages)

                    with status_placeholder.status("📟 *Generating the code*..."):
                        with st.chat_message("assistant", avatar="🤖"):
                            if not private_nvcf_endpoints:
                                stream = client.chat.completions.create(
                                    model=st.session_state["llm_model"],
                                    messages=[
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.messages
                                    ],
                                    stream=True,
                                    temperature=0.0,
                                    max_tokens=4096
                                )
                
                                result = stream_responses(input_stream=stream) 
                            else:
                                response_list = []
                                botmsg = st.empty()
                                for chunk in client.stream(
                                    [
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.messages
                                    ]
                                ):
                                    response_list.append(chunk.content)
                                    result = "".join(response_list)
                                    botmsg.write(result + "▌")
                                if result:
                                    botmsg.write(result)  

                    st.session_state.messages.append({"role": "assistant", "content": result})
                    st.session_state.messages_display.append({'role': 'assistant', 'content': result})
                    code_generated = extract_code(result)

                    with open("llm_generated_code.py", 'r') as file:

                        exec_namespace = {}
                        file_read = file.read()

                        if 'llm_response' in file_read:
                            try:
                                exec(file_read, exec_namespace)
                                llm_output_list = exec_namespace['llm_response'](dfs)
                                for llm_output in llm_output_list:
                                    if isinstance(llm_output, go.Figure):
                                        st.plotly_chart(llm_output, key=str(uuid.uuid4()))
                                        st.session_state.messages_display.append({'role': 'plot', 'figure': llm_output}) 
                                    elif isinstance(llm_output, str):
                                        result = stream_responses_static(input_stream=llm_output) 
                                        st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': llm_output})
                                    elif isinstance(llm_output, pd.DataFrame):
                                        llm_output.reset_index(drop=True, inplace=True)
                                        st.dataframe(llm_output, width=column_width)
                                        st.session_state.messages_display.append({'role': 'show_data', 'dataframe': llm_output})  
                                    else:
                                        st.write("Unknown type")
                            except Exception as e:
                                st.write(e)
                                st.write("Error has occurred during code execution. Working on the debugger LLM so that it corrects the error from the code.")

                        if share_data:
                            save_chat_history_locally(
                                data_list=st.session_state.messages, 
                                file_path='chat_history.xlsx'
                            )

                            upload_chat_history_to_s3(
                                s3_object=s3, 
                                file_path=local_file_path_chat_history, 
                                bucket=bucket_name, 
                                s3_key=s3_file_path_chat_history
                            )

                        # Clearing the LLM generated code 
                        with open("llm_generated_code.py", "w") as file:
                            pass
        st.sidebar.divider()

        if prompt := st.chat_input("Write your lines here..."):

            additional_message = f"""
            INSTRUCTIONS 
            - If it is a general purpose query, let the user only know your purpose briefly.
            - Use only one function to answer all the questions asked by the user and nothing else. ALWAYS ensure to give only llm_response(df) function.
            - Import the necessary libraries which are needed for the task for running this function. If libraries are not needed, you need not import them. 
            - Try to give as accurate and executable code as possible without syntax errors.

            a) llm_response(df) - Use this function to create a plot, insights from data and tables.
            The output should be of the list format [].
            The elements in the list can 'fig' which is a figure, 'insights' which is a string and 'table' which is a dataframe table.
            You can decide whether to use one element, two elements or all three elements in the list or more.
            You can also use more elements but ensure that they are similar in datatypes such as 'fig', 'insights' and 'table' respectively. 
            
            Be sure to use only NVIDIA colors #76B900 when plotting.

            ALWAYS use current quarter. The following is the current quarter: {fiscal_quarter}

            Here is the current month if needed: {fiscal_month}

            Here is the current fiscal year if needed: {fiscal_year}

            For bookings and zsch, use 'CRD Date Adjusted Fiscal Quarter' for quarter and other CRD Date Adjusted columns if needed
            For shippings and pulled, use 'Pl. GI Date Fiscal Quarter'
            for allocation and locked allocation, use the 'Allocation Date Fiscal Quarter' or associated Allocation Date columns
            for BUF, JFF and Supply, use the 'CRD Date Fiscal Quarter' or associated CRD Date columns i.e. CRD Date Fiscal Month etc 

            for questions about demand, be sure to consider bookings, zsch, shippings and pulled to answer your questions. 
            Do this only if they are available in 'Demand Type' category. 
            Be sure to aggregate the data if needed

            Whenever plotting, we sure to give explanation about the plot.
            Whenever giving a table, give insight about the table as well.
            Give your purpose only when needed and not without any need. 
            """

            # which is an ExcelFile format along with 'sheet_name'. The 'sheet_name' should be a keyword argument.
            enhanced_prompt = prompt + additional_message
            st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
            st.session_state.messages_display.append({'role': 'user', 'content': prompt})
            with st.chat_message("user", avatar="🔍"):
                st.markdown(prompt)

            status_placeholder = st.empty()
            error_occurred = False
            metadata = {
                'categorical_columns_dict': categorical_columns_dict,
                'numerical_columns_dict': numerical_columns_dict,
                'datetime_columns': datetime_columns,
            }

            with status_placeholder.status("📟 *Generating the code*..."):
                with st.chat_message("assistant", avatar="🤖"):
                    if not private_nvcf_endpoints:
                        stream = client.chat.completions.create(
                            model=st.session_state["llm_model"],
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                            temperature=0.0,
                            max_tokens=4096
                        )
        
                        result = stream_responses(input_stream=stream) 
                    else:
                        response_list = []
                        botmsg = st.empty()
                        for chunk in client.stream(
                            [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ]
                        ):
                            response_list.append(chunk.content)
                            result = "".join(response_list)
                            botmsg.write(result + "▌")
                        if result:
                            botmsg.write(result)  

            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.messages_display.append({'role': 'assistant', 'content': result})
            code_generated = extract_code(result)

            with open("llm_generated_code.py", 'r') as file:

                exec_namespace = {}
                file_read = file.read()

                if 'llm_response' in file_read:
                    try:
                        exec(file_read, exec_namespace)
                        llm_output_list = exec_namespace['llm_response'](dfs)
                        for llm_output in llm_output_list:
                            if isinstance(llm_output, go.Figure):
                                st.plotly_chart(llm_output, key=str(uuid.uuid4()))
                                st.session_state.messages_display.append({'role': 'plot', 'figure': llm_output}) 
                            elif isinstance(llm_output, str):
                                result = stream_responses_static(input_stream=llm_output) 
                                st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': llm_output})
                            elif isinstance(llm_output, pd.DataFrame):
                                llm_output.reset_index(drop=True, inplace=True)
                                st.dataframe(llm_output, width=column_width)
                                st.session_state.messages_display.append({'role': 'show_data', 'dataframe': llm_output})  
                            else:
                                st.write("Unknown type")
                    except Exception as e:
                        st.write(e)
                        st.write("Error has occurred during code execution. Working on the debugger LLM so that it corrects the error from the code.")
                # exec_namespace = {}
                # file_read = file.read()

                # if 'create_plot' in file_read:
                #     try:
                #         exec(file_read, exec_namespace)
                #         fig_returned = exec_namespace['create_plot'](dfs)
                #         st.plotly_chart(fig_returned)
                #         st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned}) 
                #     except Exception as e:
                #         error_occurred = True
                #         debugger_selected = Debugger(
                #             llm_model=st.session_state['debugger'],
                #             kwargs=metadata,
                #             private_nvcf_endpoints=private_nvcf_endpoints
                #         )
                #         with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                #             while error_occurred: 
                #                 error_count = 0
                #                 error_message = traceback.format_exc()
                #                 correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                #                 stream = debugger_selected.chat(
                #                     messages=correction_message,
                #                 )
                #                 result = stream_responses(input_stream=stream) 
                #                 st.session_state.messages.append(correction_message)
                #                 result_message = {"role": "assistant", "content": result}
                #                 st.session_state.messages.append(result_message)
                #                 code_generated = extract_code(result)

                #                 with open("llm_generated_code.py", 'r') as file:
                #                     exec_namespace = {}
                #                     file_read = file.read()
                #                 try:
                #                     exec(file_read, exec_namespace)
                #                     fig_returned = exec_namespace['create_plot'](dfs)
                #                     error_occurred = False
                #                 except:
                #                     error_count = error_count + 1
                #                     if error_count > debug_attempts:
                #                         error_occurred = False

                #         st.plotly_chart(fig_returned)
                #         st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned})  

                # if 'show_data' in file_read:
                #     try:
                #         exec(file_read, exec_namespace)
                #         df_generated = exec_namespace['show_data'](dfs)
                #         if color_coded_data == "Yes":
                #             styled_df = df_generated.style.applymap(color_code)
                #             st.dataframe(styled_df, width=column_width)
                #             st.session_state.messages_display.append({'role': 'show_data', 'dataframe': styled_df})   
                #         else:
                #             df_generated.reset_index(drop=True, inplace=True)
                #             st.dataframe(df_generated, width=column_width)
                #             st.session_state.messages_display.append({'role': 'show_data', 'dataframe': df_generated})   
                #     except Exception as e:
                #         error_occurred = True
                #         debugger_selected = Debugger(
                #             llm_model=st.session_state['debugger'],
                #             kwargs=metadata,
                #             private_nvcf_endpoints=private_nvcf_endpoints
                #         )
                #         with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                #             while error_occurred: 
                #                 error_count = 0
                #                 error_message = traceback.format_exc()
                #                 correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                #                 stream = debugger_selected.chat(
                #                     messages=correction_message,
                #                 )
                #                 result = stream_responses(input_stream=stream) 
                #                 st.session_state.messages.append(correction_message)
                #                 result_message = {"role": "assistant", "content": result}
                #                 st.session_state.messages.append(result_message)
                #                 code_generated = extract_code(result)

                #                 with open("llm_generated_code.py", 'r') as file:
                #                     exec_namespace = {}
                #                     file_read = file.read()
                #                 try:
                #                     exec(file_read, exec_namespace)
                #                     df_generated= exec_namespace['show_data'](dfs)
                #                     error_occurred = False
                #                 except:
                #                     error_count = error_count + 1
                #                     if error_count > debug_attempts:
                #                         error_occurred = False
                    
                #         if color_coded_data == "Yes":
                #             styled_df = df_generated.style.applymap(color_code)
                #             st.dataframe(styled_df, width=column_width)
                #             st.session_state.messages_display.append({'role': 'show_data', 'dataframe': styled_df})   
                #         else:
                #             df_generated.reset_index(drop=True, inplace=True)
                #             st.dataframe(df_generated, width=column_width)
                #             st.session_state.messages_display.append({'role': 'show_data', 'dataframe': df_generated})   

                #         # st.dataframe(df_generated, width=column_width)
                #         # st.session_state.messages_display.append({'role': 'show_data', 'dataframe': df_generated})   
                        
                # if 'data_exploration' in file_read:
                #     try:
                #         exec(file_read, exec_namespace)
                #         message_returned = exec_namespace['data_exploration'](dfs)
                #         # st.write(message_returned)

                #         if not private_nvcf_endpoints:

                #             client = OpenAI(
                #                 base_url=base_url,
                #                 api_key=api_key
                #             )

                #             stream = client.chat.completions.create(
                #                 model=st.session_state["llm_model"],
                #                 messages=[
                #                     {'role': 'user', 'content': message_returned + ". Rewrite this exactly the same except make it legible and only the statement should appear and not additional information. Note that this iS NVIDIA dataset for supply chain."}
                #                 ],
                #                 stream=True,
                #                 temperature=0.0,
                #                 max_tokens=4096
                #             )

                #             result = stream_responses(input_stream=stream) 
                #         else:
                #             response_list = []
                #             botmsg = st.empty()
                #             for chunk in client.stream(
                #                 [
                #                     {"role": m["role"], "content": m["content"]}
                #                     for m in st.session_state.messages
                #                 ]
                #             ):
                #                 response_list.append(chunk.content)
                #                 result = "".join(response_list)
                #                 botmsg.write(result + "▌")
                #             if result:
                #                 botmsg.write(result) 

                #         st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})
                #     except Exception as e:
                #         error_occurred = True
                #         debugger_selected = Debugger(
                #             llm_model=st.session_state['debugger'],
                #             kwargs=metadata,
                #             private_nvcf_endpoints=private_nvcf_endpoints
                #         )
                #         with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                #             while error_occurred: 
                #                 error_count = 0
                #                 error_message = traceback.format_exc()
                #                 correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                #                 stream = debugger_selected.chat(
                #                     messages=correction_message,
                #                 )
                #                 result = stream_responses(input_stream=stream) 
                #                 st.session_state.messages.append(correction_message)
                #                 result_message = {"role": "assistant", "content": result}
                #                 st.session_state.messages.append(result_message)
                #                 code_generated = extract_code(result)

                #                 with open("llm_generated_code.py", 'r') as file:
                #                     exec_namespace = {}
                #                     file_read = file.read()
                #                 try:
                #                     exec(file_read, exec_namespace)
                #                     message_returned = exec_namespace['data_exploration'](dfs)
                #                     error_occurred = False
                #                 except:
                #                     error_count = error_count + 1
                #                     if error_count > debug_attempts:
                #                         break
                #         if not private_nvcf_endpoints:
                #             stream = client.chat.completions.create(
                #                 model=st.session_state["llm_model"],
                #                 messages=[
                #                     {'role': 'user', 'content': message_returned + ". Rewrite this so that it is easier to understand and only the statement should appear and not additional information. Note that these are the orders for NVIDIA from various customers."}
                #                 ],
                #                 stream=True,
                #                 temperature=0.0,
                #                 max_tokens=4096
                #             )

                #             result = stream_responses(input_stream=stream) 
                #         else:
                #             response_list = []
                #             botmsg = st.empty()
                #             for chunk in client.stream(
                #                 [
                #                     {"role": m["role"], "content": m["content"]}
                #                     for m in st.session_state.messages
                #                 ]
                #             ):
                #                 response_list.append(chunk.content)
                #                 result = "".join(response_list)
                #                 botmsg.write(result + "▌")
                #             if result:
                #                 botmsg.write(result) 
                #         st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})

                # if 'generate_forecasts' in file_read:
                #     try:
                #         exec(file_read, exec_namespace)
                #         fig_returned = exec_namespace['generate_forecasts'](dfs)
                #         st.plotly_chart(fig_returned)
                #         st.session_state.messages_display.append({'role': 'forecast plot', 'figure': fig_returned}) 
                #     except Exception as e:
                #         error_occurred = True
                #         debugger_selected = Debugger(
                #             llm_model=st.session_state['debugger'],
                #             kwargs=metadata,
                #             private_nvcf_endpoints=private_nvcf_endpoints
                #         )
                #         with status_placeholder.status("🔧 *Fixing the bugs from code*..."):
                #             while error_occurred: 
                #                 error_count = 0
                #                 error_message = traceback.format_exc()
                #                 correction_message = {"role": "user", "content": f"The following error occurred for this code {code_generated}: {error_message}, give ONLY the correct code and nothing else. Ensure that there is only one output from the function."}
                #                 stream = debugger_selected.chat(
                #                     messages=correction_message,
                #                 )
                #                 result = stream_responses(input_stream=stream) 
                #                 st.session_state.messages.append(correction_message)
                #                 result_message = {"role": "assistant", "content": result}
                #                 st.session_state.messages.append(result_message)
                #                 code_generated = extract_code(result)

                #                 with open("llm_generated_code.py", 'r') as file:
                #                     exec_namespace = {}
                #                     file_read = file.read()
                #                 try:
                #                     exec(file_read, exec_namespace)
                #                     fig_returned = exec_namespace['generate_forecasts'](dfs)
                #                     error_occurred = False
                #                 except:
                #                     error_count = error_count + 1
                #                     if error_count > debug_attempts:
                #                         error_occurred = False

                #         st.plotly_chart(fig_returned)
                #         st.session_state.messages_display.append({'role': 'forecast plot', 'figure': fig_returned})

                # if 'general_purpose' in file_read:

                #     exec(file_read, exec_namespace)
                #     message_returned = exec_namespace['general_purpose']()

                #     if not private_nvcf_endpoints:
                #         client = OpenAI(
                #             base_url=base_url,
                #             api_key=api_key
                #         )

                #         stream = client.chat.completions.create(
                #             model=st.session_state["llm_model"],
                #             messages=[
                #                 {'role': 'user', 'content': f"The following is the input message given by user: {prompt}" + ".\n" + f"The following is the reply by the large language model: {message_returned}" + ". Rewrite only the reply for this by the large language model while keeping the context of the question and tell this as if you were speaking to demand planner from Nvidia. If the message is outside the scope of the NVIDIA dataset, be sure to let them know your purpose and not answer question about the user input and say sorry but be sure to always be polite and very respectful. Do not use quotes."}
                #             ],
                #             stream=True,
                #             temperature=0.0,
                #             max_tokens=4096
                #         )

                #         result = stream_responses(input_stream=stream) 
                #     else:
                #         response_list = []
                #         botmsg = st.empty()
                #         for chunk in client.stream(
                #             [
                #                 {"role": m["role"], "content": m["content"]}
                #                 for m in st.session_state.messages
                #             ]
                #         ):
                #             response_list.append(chunk.content)
                #             result = "".join(response_list)
                #             botmsg.write(result + "▌")
                #         if result:
                #             botmsg.write(result) 

                #     st.session_state.messages_display.append({'role': 'adhoc', 'message from adhoc query': result})

                if share_data:
                    save_chat_history_locally(
                        data_list=st.session_state.messages, 
                        file_path='chat_history.xlsx'
                    )

                    upload_chat_history_to_s3(
                        s3_object=s3, 
                        file_path=local_file_path_chat_history, 
                        bucket=bucket_name, 
                        s3_key=s3_file_path_chat_history
                    )

                # Clearing the LLM generated code 
                with open("llm_generated_code.py", "w") as file:
                    pass

if __name__ == "__main__":

    st.set_page_config(
        page_title="Demand Planner Co-Pilot",
        page_icon="https://www.nvidia.com/favicon.ico"
    )

    st.markdown("""
        <style>
        .title {
            color: #76B900;
            font-size: 2.5em;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">Demand Planner Co-Pilot 📊</h1>', unsafe_allow_html=True)
    st.markdown("**Prompt about your data, and get actionable insights (*check for accuracy*)** ✨")
    content_screen()

    # auth_provider = AzureADOAuthProvider()

    # # Session and local storage check
    # if "logged_in" not in st.session_state:
    #     st.session_state.logged_in = False
    # if "token" not in st.session_state:
    #     st.session_state.token = None
    # if "logout_clicked" not in st.session_state:
    #     st.session_state.logout_clicked = False

    # authorize_url = f"{auth_provider.get_authorize_url()}&prompt=login"

    # # Handle login if the authorization code is in query params and user is not logged in
    # if "code" in st.query_params and not st.session_state.logged_in:
    #     code = st.query_params.get("code")
    #     try:
    #         st.session_state.token = auth_provider.get_token(code)
    #         st.query_params.clear()
    #         user_info = auth_provider.get_user_info(st.session_state.token)
    #         st.session_state.logged_in = True
    #         # local_storage.setItem("logged_in", "true", key="login_status_key")
    #         # local_storage.setItem("token", st.session_state.token, key="token_key")
    #     except Exception as e:
    #         st.error(f"An error occurred: {e}")

    # # Display logged-in status and user info
    # if st.session_state.logged_in:
    #     try:
    #         user_info = auth_provider.get_user_info(st.session_state.token)
    #     except:
    #         # Clear session state and local storage on logout
    #         st.session_state.clear()
    #         st.session_state.logout_clicked = True  # Set logout flag
    #         st.rerun()  # First rerun to clear session and storage

    #     dl_user_allowed = check_user_eligibility(
    #         user_info=user_info,
    #         dls_list=["anoguera-staff"],
    #     )
    #     if dl_user_allowed:
    #         content_screen()
    #     else:
    #         with st.chat_message("assistant", avatar="🤖"):
    #             st.markdown("Please request access to the application using the DL address provided. Thank you.")
    #             st.markdown("If you have been provided DL access, please ensure to turn on VPN before signing.")

    #     if st.sidebar.button(":material/logout: Logout", key="logout_button"):
    #         st.session_state.clear()

    #         st.session_state.logout_clicked = True 
    #         st.rerun()  # First rerun to clear session and storage

    # # Extra check to enforce logout on rerun
    # if st.session_state.get("logout_clicked"):
    #     st.session_state.logout_clicked = False  # Reset the flag
    #     st.rerun()  # Second rerun to refresh the interface fully

    # # Show login button if the user is not logged in
    # elif not st.session_state.logged_in:
    #     if st.button(":material/login: Login to NVIDIA", key="login_button"):
    #         st.write("Redirecting to Microsoft login...")
    #         st.markdown(f"<meta http-equiv='refresh' content='0;url={authorize_url}'>", unsafe_allow_html=True)


   



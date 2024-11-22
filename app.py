
from openai import OpenAI
import streamlit as st
import re
import traceback
import pandas as pd
import numpy as np
import time
from src.debugger import Debugger
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

import streamlit as st

# Getting access to credentials for loading data to database
platform = st.secrets['DATABASE']['PLATFORM']
aws_access_key_id = st.secrets['DATABASE']['AWS_ACCESS_KEY_ID']
aws_secret_access_key = st.secrets['DATABASE']['AWS_SECRET_ACCESS_KEY']
region_name = st.secrets['DATABASE']['REGION_NAME']
memory_location = st.secrets['DATABASE']['BUCKET']
number = st.secrets['DATABASE']['NUMBER']

# Getting access to user credentials to validate login
username_credentials = st.secrets['USER CREDENTIALS']['USERNAME']
password_credentials = st.secrets['USER CREDENTIALS']['PASSWORD']

base_url = st.secrets['USER CREDENTIALS']['BASE_URL']
api_key = st.secrets['USER CREDENTIALS']['API_KEY']

# Getting access to NVCF credentials for inference
NVCF_CHAT_FUNCTION_ID = st.secrets['NVCF CREDENTIALS']['NVCF_CHAT_FUNCTION_ID']
NVCF_URL = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{NVCF_CHAT_FUNCTION_ID}"
NVCF_API_KEY = st.secrets['NVCF CREDENTIALS']['NVCF_API_KEY']
MODEL = "meta/llama-3.1-8b-instruct"
os.environ['NVIDIA_API_KEY'] = NVCF_API_KEY

st.set_page_config(
    page_title="Demand Planner Co-Pilot",
    page_icon="https://www.nvidia.com/favicon.ico",
    initial_sidebar_state="collapsed"
)

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
        st.success(f"Thank you for the feedback")
    except Exception as e:
        st.error(f"Error uploading file: {e}")

# Encapsulated function to upload feedback to S3
def upload_chat_history_to_s3(s3_object, file_path, bucket, s3_key):
    try:
        s3_object.upload_file(file_path, bucket, s3_key)
    except Exception as e:
        st.error(f"Error uploading file: {e}")

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

# First Page: Login Page
def login_page():

    # Inject CSS to change title color
    st.markdown("""
        <style>
        .title {
            color: #76B900;
            font-size: 2.5em;
        }
        </style>
        """, unsafe_allow_html=True)

    # Add your title with a specific CSS class
    st.markdown('<h1 class="title">Demand Planner Co-Pilot 📊</h1>', unsafe_allow_html=True)

    # Wrap the entire form in one large box
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    
    # Create a form inside the single box
    with st.form("login_form"):
        st.header("Login to your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Check login on form submit
    if submit_button:
        if username == username_credentials and password == password_credentials:
            st.session_state['logged_in'] = True
            st.rerun()  # Redirect to the next page immediately
        else:
            st.warning("Invalid username or password")

def task_page():
    # Inject CSS to change title color
    st.markdown("""
        <style>
        .title {
            color: #76B900;
            font-size: 2.5em;
        }
        </style>
        """, unsafe_allow_html=True)

    # Add your title with a specific CSS class
    st.markdown('<h1 class="title">Demand Planner Co-Pilot 📊</h1>', unsafe_allow_html=True)

    st.markdown("**Prompt about your data, and get actionable insights (*check for accuracy*)** ✨")

    # Wrap the entire form in one large box
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    
    # Create a form inside the single box
    with st.form("login_form"):
        # st.header("Select the task:")

        # logo = Image.open("images/NVIDIA-logo-BL.jpg")
        # st.image(logo, width=500)
        # help_box = """
        # Analyze single data source - Analyzes a single snapshot file for insights and advanced analytics.\n
        # Compare changes between data sources - Compares two snapshot files for changes, insights and advanced analytics.
        # """
        # task_selected = st.radio("Select the task:", ["Analyze single data source", "Compare changes between data sources"],
        #                                           help=help_box)
        
        help_chat_history = """
        Do you want to allow sharing history of conversations?

        Yes - The chat conversations are recorded for improvement in answer quality, content and code generation.\n
        No - The chat conversations are not recorded."""
        share_chat_history = st.radio("Share chats for product development?", ["Yes", "No"], help=help_chat_history, index=1)
        
        help_display = """
        Displays the colors for numbers based on whether they are positive or negative for the generated tables.
        """

        color_coded_data = st.radio("Display colored dataset:", ["Yes", "No"], index=1, help=help_display)

        # share_data = st.checkbox("Share chats for product development?")
        share_data = st.session_state['share_chat_history_button']

        # st.session_state['task_selected_state'] = task_selected
        st.session_state['share_chat_history_button'] = share_chat_history
        st.session_state['display_colored_state'] = color_coded_data


        submit_button = st.form_submit_button(label="Submit")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Check login on form submit
    if submit_button:
        st.session_state['task_submitted'] = True
        st.rerun()

# Second Page: Content Page
def content_page():

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

    # Clearing the LLM generated code 
    with open("llm_generated_code.py", "w") as file:
        pass

    FLAG = 0
            
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
                return ''  # No styling applied for values equal to the threshold
        else:
            return ''  # No styling for non-numeric values
        
    # Inject CSS to change title color
    st.markdown("""
        <style>
        .title {
            color: #76B900;
            font-size: 2.5em;
        }
        </style>
        """, unsafe_allow_html=True)

    # Add your title with a specific CSS class
    st.markdown('<h1 class="title">Demand Planner Co-Pilot 📊</h1>', unsafe_allow_html=True)

    st.markdown("**Prompt about your data, and get actionable insights (*check for accuracy*)** ✨")

    tab_selected = st.sidebar.tabs(["Configuration", "Chat History", "Feedback"])

    with tab_selected[0]:

        private_nvcf_endpoints = st.toggle("Private NVCF Endpoints", value=True)
        if not private_nvcf_endpoints:
            model_selected_help = """
            The following are the large language models available: 

            1. Llama-3.1 405b - Useful when the results should be accurate
            2. Llama-3.1 70b - Useful when the results generated are fast
            """
            model_selected = st.selectbox("Select the large language model:", ["Llama-3.1 70b", "Llama-3.1 405b"], help=model_selected_help)
            if "llm_model" not in st.session_state:
                st.session_state["llm_model"] = []

            if "debugger" not in st.session_state:
                st.session_state["debugger"] = []

            # if model_selected == "Llama-3.1 8b":
            #     st.session_state["llm_model"] = "meta/llama-3.1-8b-instruct"
            #     st.session_state["debugger"] = "meta/llama-3.1-8b-instruct"
            if model_selected == "Llama-3.1 70b":
                st.session_state["llm_model"] = "meta/llama-3.1-70b-instruct"
                st.session_state["debugger"] = "meta/llama-3.1-70b-instruct"
            else:
                st.session_state["llm_model"] = "meta/llama-3.1-405b-instruct"
                st.session_state["debugger"] = "meta/llama-3.1-405b-instruct"

            client = OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
        else:
            register_model(Model(id=MODEL, model_type="chat", client="ChatNVIDIA", endpoint=NVCF_URL))
            client = ChatNVIDIA(
                model=MODEL,
                temperature=0.0,
                max_tokens=4096
            )
            if "debugger" not in st.session_state:
                st.session_state["debugger"] = []
            st.session_state["llm_model"] = MODEL

        # task_selected = st.session_state['task_selected_state']
        # color_coded_data = st.radio("Display colored dataset:", ["Yes", "No"], index=1, help="Displays the colors based on positive and negative numbers")
        task_selected = "Analyze single data source"
        column_width = st.slider("Adjust displayed column width:", 100, 1000, value=750, help="Adjust the width of the generated tables.")
        debug_attempts = st.slider("Select the maximum code retries:", 1, 5, value=3, help="Adjust the total attempts by LLM during code generation failed attempts.")
        # share_data = st.checkbox("Share chats for product development?")
        share_data = st.session_state['share_chat_history_button']
        color_coded_data = st.session_state['display_colored_state']
        st.sidebar.markdown("*v2024.9.1 beta release*")

    with tab_selected[2]:
        st.header("Feedback")
        feedback = st.text_area("Please provide your feedback here:", height=150)
        if st.button("Submit", help="Click to submit feedback"):
            if feedback:            
                save_feedback_locally(feedback, local_file_path_feedback)
                upload_feedback_to_s3(
                    s3_object=s3,
                    file_path=local_file_path_feedback,
                    bucket=bucket_name, 
                    s3_key=s3_file_path_feedback
                )

    @st.cache
    def load_data(file, sheet_name):
        return pd.read_excel(file, sheet_name=sheet_name)
    
    @st.cache_data
    def load_csv_aws(file):
        response = s3.get_object(Bucket=bucket_name, Key=f'DP Allocation File/{file}')
        csv_content = response['Body'].read().decode('utf-8')
        output = pd.read_csv(StringIO(csv_content))

        return output
    
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

        datetime_columns = ['Ord Date', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date', 'CRD Date']
        categorical_columns = {}
        for col in dfs.columns:
            if col not in datetime_columns:
                if dfs[col].dtype == "object":
                    categorical_columns[col] = list(dfs[col].unique())
        return categorical_columns
    
    @st.cache_data
    def compare_categorical_columns(df1, df2):
        # Specify the datetime columns to exclude
        datetime_columns = ['Ord Date', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date', 'CRD Date']
        
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
        actual_datetime_columns = ['Ord Date', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date', 'CRD Date']
        for col in dfs.columns:
            if col in actual_datetime_columns:
                datetime_columns.append(col)
            # if dfs[col].dtype == "<M8[ns]":
            #     datetime_columns.append(col)
        return datetime_columns

    status_placeholder = st.empty()
    error_occurred = False



    if task_selected == "Compare changes between data sources":

        
        with st.expander("Click to view the data", expanded=True):

            flag = 0

            col1, col2 = st.columns(2)

            files_list = get_aws_files()

            with col1:
                # initial_date = st.date_input("Initial Data Source Date:", value=datetime(2024, 6, 17))
                # initial_date_string = initial_date.strftime("%Y-%m-%d")
                # if initial_date_string == '2024-06-17':
                #     dfs = load_excel_file("data/CustomerAllocation20240617.xlsx")

                first_file = st.selectbox("Select the file to analyze:", files_list)
                updated_files_list = files_list.copy()
                # response = s3.get_object(Bucket=bucket_name, Key=f'DP Allocation File/{first_file}')
                # csv_content = response['Body'].read().decode('utf-8')
                # dfs = pd.read_csv(StringIO(csv_content))
                dfs = load_csv_aws(file=first_file)
                dfs = divide_data(data=dfs, divider=number)


            with col2:
                updated_files_list.remove(first_file)
                second_file = st.selectbox("Select the second file to analyze:", updated_files_list)
                dfs2 = load_csv_aws(file=second_file)
                dfs2 = divide_data(data=dfs2, divider=number)
                # response = s3.get_object(Bucket=bucket_name, Key=f'DP Allocation File/{second_file}')
                # csv_content = response['Body'].read().decode('utf-8')
                # dfs2 = pd.read_csv(StringIO(csv_content))
                # dfs2 = pd.read_csv(second_file)
                # final_date = st.date_input("Final Data Source Date:", value=datetime(2024, 6, 24))
                # final_date_string = final_date.strftime("%Y-%m-%d")
                # if final_date_string == '2024-06-24':
                #     dfs2 = load_excel_file("data/CustomerAllocation20240624.xlsx")

            # if initial_date_string == '2024-06-17' and final_date_string == '2024-06-24':
            #     pass
            # else:
            #     dfs = load_excel_file("data/CustomerAllocation20240617.xlsx")
            #     dfs2 = load_excel_file("data/CustomerAllocation20240624.xlsx")
            #     st.warning(f"Note: The data source is loaded only for '2024-06-17' and '2024-06-24' dates.")

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

            demand_type_selected = st.multiselect("Select the Demand Type:", dfs['Demand Type'].unique())

            if demand_type_selected:

                flag = 1

                dfs = dfs[dfs['Demand Type'].isin(demand_type_selected)]
                dfs2 = dfs2[dfs2['Demand Type'].isin(demand_type_selected)]

                empty_columns = dfs.columns[dfs.isna().all()]
                empty_columns2 = dfs2.columns[dfs2.isna().all()]
                dfs.drop(columns=empty_columns, inplace=True)
                dfs2.drop(columns=empty_columns2, inplace=True)

                dfs.reset_index(drop=True, inplace=True)
                st.write(f"First file selected: {first_file}")
                st.dataframe(dfs, height=200)

                dfs2.reset_index(drop=True, inplace=True)
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


            system_message_file_uploaded = f"""
                If it is a general purpose query, NEVER follow anything present in triple hiphens. 
                Instead reply like a general purpose large language model without any coding.
                
                ---
                The following in double backticks (``) is provided for you to get context about the dataset.
                Do not use this information for plotting but only to help you get understanding of the dataset.
                ``
                The following are the categorical columns and and the corresponding unique categories per column present in the first file only.
                
                This is the name of the initial file to analyze which is a csv: {first_file}

                <{dfs_cat}>

                The following are the numerical columns present in the data.
            
                <{numerical_columns_dict}>

                The following are the datetime columns present in the data.

                <{datetime_columns}>

                The following are the categorical columns and and the corresponding unique categories per column present in the second file only.

                This is the name of the second file to analyze which is a csv: {second_file}

                <{dfs2_cat}>

                The following are the numerical columns present in the data.
            
                <{numerical_columns_dict2}>

                The following are the datetime columns present in the data.

                <{datetime_columns2}>

                The following are the intersection unique values for the columns and the unique values between the two files:

                <{intersect_cat}>

                ``

                The following in double slashes give the description and meaning of each of the columns in the data
                \\
                Demand Type - This gives information about the type of demand which could be allocation, supply, demand, regional sales forecast, bottom up forecast, judged field forecast and others
                Ord Type - It is about whether it is internal demand or external demand 
                Cust Name - This is the name of the customer who has ordered the products but not the end customer
                Region - the region from which the order has taken place
                End Cust Name - This is the final end customer who made purchase of the products 
                Quantity - This meaning of this column is different based on the demand type selected. For booking, it gives booking quantities and so on
                Amount - This is the dollar amount price set for the products
                Ord Date - This is the date at which there were customer orders for products
                CRD Date - This is the customer request date for various products or materials 
                Inv Date - This is the invoice date given after customer makes a purchase and if the material is available
                PI. GI Date - This is the actual shipping date for the product SKUs
                MAD - This stands for material availability date which determines when the material is actually available
                Ship Plnt - This is the shipping plant from which the product SKU is present
                Business Unit - This indicates various business units from NVIDIA
                Types - This is similar to business unit 
                Family - This gives idea about the product family 
                Allocation Date - This is the date at which a particular material is allocated to a specific customer
                Allocation Qty - This is the amount of allocation given by NVIDIA to various customers. Note that when allocation quantityt is present,
                it means that Quantity will be None
                \\

                Use any of these packages Pandas, Streamlit and Plotly ONLY. 
                Provide SINGLE CODE BLOCK when needed.

                INSTRUCTIONS
                - When user gives additional queries, ALWAYS give the FULL and COMPLETE code.
                - ALWAYS give functions and inside them, give code.
                - USE SINGLE CODE BLOCK with a solution 
                - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK
                - The code block must start and end with ```python and then end with ```
                - Import ALL necessary modules when executing code 
                - Use different colors when used to plot
                ---


                Here are examples of user queries and how your response should look like:

                User: Give a table to show for each SKU, the initial quantity, final quantity, change per SKU

                Assistant Response:
                ```python
                import pandas as pd

                def show_diff_data(dfs, dfs2):
                    df1 = dfs[['SKU', 'Quantity']].groupby('SKU').sum().reset_index()
                    df2 = dfs2[['SKU', 'Quantity']].groupby('SKU').sum().reset_index()

                    df_generated = pd.merge(df1, df2, on='SKU', how='outer', suffixes=('_initial', '_final'))
                    df_generated = df_generated.replace(np.nan, 0)
                    df_generated['change'] = df_generated['Quantity_final'] - df_generated['Quantity_initial']

                    return df_generated
                ```

                User: Give a plot to show the demand for SKU A from the first file

                Assistant Response:
                ```python
                import pandas as pd
                import plotly.express as px

                def plot_changes(dfs, dfs2):
                    # Filter data for A
                    filtered_df = dfs[(dfs['SKU'] == 'A') & (dfs['Demand Type'] == 'BOOKED')]

                    # Convert CRD Date to datetime
                    filtered_df['CRD Date'] = pd.to_datetime(filtered_df['CRD Date'])

                    # Aggregate demand based on CRD date
                    aggregated_df = filtered_df.groupby('CRD Date')['Quantity'].sum().reset_index()

                    # Plot the demand
                    fig = px.line(aggregated_df, x='CRD Date', y='Quantity', title='Demand for SKU A', color_discrete_sequence=['#76B900'])
                    return fig
                ```

                User: What were the changes in demand for product SKU A?

                Assistant Response:
                ```python
                import pandas as pd

                def data_difference(dfs, dfs2):
                    df1 = dfs[(dfs['Demand Type'] == 'BOOKED') & (dfs['SKU'] == 'A') & (dfs['Business Unit'].notna())][['Business Unit', 'Quantity']].groupby('Business Unit').sum().reset_index()
                    df2 = dfs2[(dfs2['Demand Type'] == 'BOOKED') & (dfs2['SKU'] == 'A') & (dfs2['Business Unit'].notna())][['Business Unit', 'Quantity']].groupby('Business Unit').sum().reset_index()

                    df_generated = pd.merge(df1, df2, on='Business Unit', how='outer', suffixes=('_initial', '_final'))
                    df_generated = df_generated.replace(pd.NA, 0)
                    df_generated['change'] = df_generated['Quantity_final'] - df_generated['Quantity_initial']

                    return "The changes per SKU for product A are: \n" + str(df_generated)
                ```

                User:
            """

            # system_message_file_uploaded = f"""
            #     If it is a general purpose query, NEVER follow anything present in triple hiphens. 
            #     Instead reply like a general purpose large language model without any coding.
                
            #     ---
            #     The following in double backticks (``) is provided for you to get context about the dataset.
            #     Do not use this information for plotting but only to help you get understanding of the dataset.
            #     ``
            #     The following are the categorical columns and the corresponding unique categories per column.
                
            #     This is the name of the initial file to analyze which is a csv: {first_file}

            #     <{categorical_columns_dict}>

            #     The following are the numerical columns present in the data.
            
            #     <{numerical_columns_dict}>

            #     The following are the datetime columns present in the data.

            #     <{datetime_columns}>

            #     This is the name of the second file to analyze which is a csv: {second_file}

            #     <{categorical_columns_dict2}>

            #     The following are the numerical columns present in the data.
            
            #     <{numerical_columns_dict2}>

            #     The following are the datetime columns present in the data.

            #     <{datetime_columns2}>

            #     ``

            #     The following in double slashes give the description and meaning of each of the columns in the data
            #     \\
            #     Demand Type - This gives information about the type of demand which could be allocation, supply, demand, regional sales forecast, bottom up forecast, judged field forecast and others
            #     Ord Type - It is about whether it is internal demand or external demand 
            #     Cust Name - This is the name of the customer who has ordered the products but not the end customer
            #     Region - the region from which the order has taken place
            #     End Cust Name - This is the final end customer who made purchase of the products 
            #     Quantity - This meaning of this column is different based on the demand type selected. For booking, it gives booking quantities and so on
            #     Amount - This is the dollar amount price set for the products
            #     Ord Date - This is the date at which there were customer orders for products
            #     CRD Date - This is the customer request date for various products or materials 
            #     Inv Date - This is the invoice date given after customer makes a purchase and if the material is available
            #     PI. GI Date - This is the actual shipping date for the product SKUs
            #     MAD - This stands for material availability date which determines when the material is actually available
            #     Ship Plnt - This is the shipping plant from which the product SKU is present
            #     Business Unit - This indicates various business units from NVIDIA
            #     Types - This is similar to business unit 
            #     Family - This gives idea about the product family 
            #     Allocation Date - This is the date at which a particular material is allocated to a specific customer
            #     Allocation Qty - This is the amount of allocation given by NVIDIA to various customers. Note that when allocation quantityt is present,
            #     it means that Quantity will be None
            #     \\

            #     Use any of these packages Pandas, Streamlit and Plotly ONLY. 
            #     Provide SINGLE CODE BLOCK when needed.

            #     INSTRUCTIONS
            #     - When user gives additional queries, ALWAYS give the FULL and COMPLETE code.
            #     - ALWAYS give functions and inside them, give code.
            #     - USE SINGLE CODE BLOCK with a solution 
            #     - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK
            #     - The code block must start and end with ```python and then end with ```
            #     - Import ALL necessary modules when executing code 
            #     - Use different colors when used to plot
            #     ---


            #     Here are examples of user queries and how your response should look like:

            #     User: Give a table to show for each SKU, the initial quantity, final quantity, change per SKU

            #     Assistant Response:
            #     ```python
            #     import pandas as pd

            #     def show_diff_data(dfs, dfs2):
            #         df1 = dfs[['SKU', 'Quantity']].groupby('SKU').sum().reset_index()
            #         df2 = dfs2[['SKU', 'Quantity']].groupby('SKU').sum().reset_index()

            #         df_generated = pd.merge(df1, df2, on='SKU', how='outer', suffixes=('_initial', '_final'))
            #         df_generated = df_generated.replace(np.nan, 0)
            #         df_generated['change'] = df_generated['Quantity_final'] - df_generated['Quantity_initial']

            #         return df_generated
            #     ```

            #     User: Give a plot to show the demand for SKU A from the first file

            #     Assistant Response:
            #     ```python
            #     import pandas as pd
            #     import plotly.express as px

            #     def plot_changes(dfs, dfs2):
            #         # Filter data for A
            #         filtered_df = dfs[(dfs['SKU'] == 'A') & (dfs['Demand Type'] == 'BOOKED')]

            #         # Convert CRD Date to datetime
            #         filtered_df['CRD Date'] = pd.to_datetime(filtered_df['CRD Date'])

            #         # Aggregate demand based on CRD date
            #         aggregated_df = filtered_df.groupby('CRD Date')['Quantity'].sum().reset_index()

            #         # Plot the demand
            #         fig = px.line(aggregated_df, x='CRD Date', y='Quantity', title='Demand for SKU A', color_discrete_sequence=['#76B900'])
            #         return fig
            #     ```

            #     User: What were the changes in demand for product SKU A?

            #     Assistant Response:
            #     ```python
            #     import pandas as pd

            #     def data_difference(dfs, dfs2):
            #         df1 = dfs[(dfs['Demand Type'] == 'BOOKED') & (dfs['SKU'] == 'A') & (dfs['Business Unit'].notna())][['Business Unit', 'Quantity']].groupby('Business Unit').sum().reset_index()
            #         df2 = dfs2[(dfs2['Demand Type'] == 'BOOKED') & (dfs2['SKU'] == 'A') & (dfs2['Business Unit'].notna())][['Business Unit', 'Quantity']].groupby('Business Unit').sum().reset_index()

            #         df_generated = pd.merge(df1, df2, on='Business Unit', how='outer', suffixes=('_initial', '_final'))
            #         df_generated = df_generated.replace(pd.NA, 0)
            #         df_generated['change'] = df_generated['Quantity_final'] - df_generated['Quantity_initial']

            #         return "The changes per SKU for product A are: \n" + str(df_generated)
            # """

        if "messages" not in st.session_state:

            st.session_state.messages = ["placeholder"]

        st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}

        # with tab_selected[1]:
        #     try:
        #         for selected_conversation in list(st.session_state.conversations.keys()):
        #             if st.button(selected_conversation):
        #                 conversation_selected = selected_conversation
        #                 st.session_state.button_clicked_flag = True
        #     except:
        #         conversation_selected = selected_conversation
        #     clear_chat_history = st.button("New chat", help="click for new chat")

        # if clear_chat_history:
        #     st.session_state.flag_new_chat = 1
        #     now = datetime.now()
        #     st.session_state.new_formatted_date_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        #     st.session_state.conversations[f"{st.session_state.new_formatted_date_time}"] = []
        #     st.session_state.messages = []
        #     st.session_state.messages_display = []
        #     st.session_state.debugger_messages = []
        #     st.session_state.messages = ["placeholder"]
        #     st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}

        # if st.session_state.flag_new_chat == 0:
        #     st.session_state.conversations[f'{st.session_state.old_formatted_date_time}'] = st.session_state.messages_display
        # else:
        #     if st.session_state.button_clicked_flag == True:
        #         st.session_state.conversations[conversation_selected] = st.session_state.messages_display
        #     else:
        #         st.session_state.conversations[f'{st.session_state.new_formatted_date_time}'] = st.session_state.messages_display
    
        # st.session_state.button_clicked_flag = False

        with tab_selected[1]:

            if st.button("📝 Click for new chat", help="Click for new chat"):
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


    

        i = 0

        # try:
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
        # except:
        #     conversation_selected = st.session_state.old_formatted_date_time
        #     for message in st.session_state.conversations[conversation_selected]:
        #         if message['role'] == 'user':
        #             with st.chat_message(message['role'], avatar="🔍"):
        #                 st.markdown(message['content'])
        #         if message['role'] == 'assistant':
        #             with st.status("📟 *Generating the code*..."):
        #                 with st.chat_message(message['role'], avatar="🤖"):
        #                     st.markdown(message['content'])
        #         if message['role'] == 'plot':
        #             st.plotly_chart(message['figure'])
        #         if message['role'] == 'adhoc':
        #             st.write(message['message from adhoc query'])
        #         if message['role'] == 'show_diff_data':
        #             st.dataframe(message['dataframe'], width=column_width)


        if flag == 1:    
            if prompt := st.chat_input("Write your lines here..."):
                additional_message = f"""
                INSTRUCTIONS 
                - If it is a general purpose query, NEVER give a reply but reply saying that you are ready to assist with the dataset.
                - Import the necessary libraries which are needed for the task.
                - Only use one or more of these functions and do not write code outside of the functions. The output 
                should be only the functions.
                - Try to give as accurate and executable code as possible without syntax errors.

                a) data_difference(dfs, dfs2)
                b) show_diff_data(dfs, dfs2)
                c) plot_changes(dfs, dfs2)
                d) general_purpose()
                
                a) data_difference(dfs, dfs2) - If asked to find changes in data or comparison between datasets, use this function. The output could either be insights from changes as a single return output. The insights should be a string.
                b) show_diff_data(dfs, dfs2) - You can also use 'show_diff_data' function to return a single dataframe called 'df_generated'. The output should be one argument. The input should be multiple dataframes 'dfs' and 'dfs2'.
                c) plot_changes(dfs, dfs2) - if asked to plot changes in data or comparison between datasets, use this function which accepts dfs and dfs2 as arguments. The output could either be plots of the insights from changes as a single return output. When using this, ensure to give proper titles and also necessary labels for plots.
                d) general_purpose() - Use this function to answer general purpose questions very briefly but reminding about your purpose of analyzing the data and providing advanced analytics for Nvidia dataset. Return a string.

                The following are the available demand types from the data: {categorical_columns_dict['Demand Type']}
                Be sure to always filter the data based on these demand types. 
                If the question is about allocation or locked allocation and when the demand type present in the data has 'Allocation' or 'Locked Allocation', 
                ensure to look for 'Allocation Qty' for quantities and dates as 'Allocation Date'.

                Try to use NVIDIA colors where it makes sense when plotting.
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
        files_list = get_aws_files()

        with st.expander("Click to view the data", expanded=True):

            flag = 0

            # initial_date = st.date_input("Data Source Date:", value=datetime(2024, 6, 17))
            # initial_date_string = initial_date.strftime("%Y-%m-%d")
            # if initial_date_string == '2024-06-17':
            #     dfs = load_excel_file("data/CustomerAllocation20240617.xlsx")
            # else:
            #     dfs = load_excel_file("data/CustomerAllocation20240617.xlsx")
            #     st.warning(f"Note: The data source is loaded only for '2024-06-17' date.")

            first_file = st.selectbox("Select the file to analyze:", files_list)
            dfs = load_csv_aws(file=first_file)
            dfs = divide_data(data=dfs, divider=number)

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

            demand_type_selected = st.multiselect("Select the Demand Type:", dfs['Demand Type'].unique())

            if demand_type_selected:

                flag = 1
                dfs = dfs[dfs['Demand Type'].isin(demand_type_selected)]

                empty_columns = dfs.columns[dfs.isna().all()]
                dfs.drop(columns=empty_columns, inplace=True)

                dfs.reset_index(drop=True, inplace=True)
                st.write(f"File selected: {first_file}")
                st.write(dfs)

            # Get and display categorical columns
            categorical_columns_dict = get_categorical_columns(dfs)

            # Get and display numerical columns
            numerical_columns_dict = get_numerical_columns(dfs)

            datetime_columns = get_datetime_columns(dfs)

            for datetime_column in datetime_columns:
                dfs[datetime_column] = pd.to_datetime(dfs[datetime_column])

            system_message_file_uploaded = f"""
                If it is a general purpose query, NEVER follow anything present in triple hiphens. 
                Instead reply like a general purpose large language model without any coding.
                
                ---
                The following in double backticks (``) is provided for you to get context about the dataset.
                Do not use this information for plotting but only to help you get understanding of the dataset.
                ``
                The following are the categorical columns and the corresponding unique categories per column.
                
                This is the csv file which is selected: {first_file}

                <{categorical_columns_dict}>

                The following are the numerical columns present in the data.
            
                <{numerical_columns_dict}>

                The following are the datetime columns present in the data.

                <{datetime_columns}>

                ``

                The following in double slashes give the description and meaning of each of the columns in the data
                \\
                Demand Type - This gives information about the type of demand which could be allocation, supply, demand, regional sales forecast, bottom up forecast, judged field forecast and others
                Ord Type - It is about whether it is internal demand or external demand 
                Cust Name - This is the name of the customer who has ordered the products but not the end customer
                Region - the region from which the order has taken place
                End Cust Name - This is the final end customer who made purchase of the products 
                Quantity - This meaning of this column is different based on the demand type selected. For booking, it gives booking quantities and so on
                Amount - This is the dollar amount price set for the products
                Ord Date - This is the date at which there were customer orders for products
                CRD Date - This is the customer request date for various products or materials 
                Inv Date - This is the invoice date given after customer makes a purchase and if the material is available
                PI. GI Date - This is the actual shipping date for the product SKUs
                MAD - This stands for material availability date which determines when the material is actually available
                Ship Plnt - This is the shipping plant from which the product SKU is present
                Business Unit - This indicates various business units from NVIDIA
                Types - This is similar to business unit 
                Family - This gives idea about the product family 
                Allocation Date - This is the date at which a particular material is allocated to a specific customer
                Allocation Qty - This is the amount of allocation given by NVIDIA to various customers. Note that when allocation quantityt is present,
                it means that Quantity will be None
                \\

                Use any of these packages Pandas, Streamlit and Plotly ONLY. 
                Provide SINGLE CODE BLOCK when needed.

                INSTRUCTIONS
                - When user gives additional queries, ALWAYS give the FULL and COMPLETE code.
                - ALWAYS give functions and inside them, give code.
                - USE SINGLE CODE BLOCK with a solution 
                - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK
                - The code block must start and end with ```python and then end with ```
                - Import ALL necessary modules when executing code 
                - Use different colors when used to plot
                ---

                Here are examples of user queries and how your response should look like:

                User: what's the allocation of 'Product A' in May

                Assistant Response:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Filter data for Product A in May
                    filtered_df = df[(df['SKU'] == 'Product A') & (df['Allocation Date'].dt.month == 5)]

                    # Get the allocation quantity
                    allocation_qty = filtered_df['Allocation Qty'].sum()

                    return f"The allocation of Product A in May is {{allocation_qty}}"
                ```

                User: What is the allocation for WS?

                Assistant Response:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Filter data for WS
                    filtered_df = df[df['Business Unit'] == 'WS']

                    # Get the allocation quantity
                    allocation_qty = filtered_df['Allocation Qty'].sum()

                    return f"The total allocation for WS is {{allocation_qty}}"
                ```

                User: What is meant by operations research?

                Assistant Response:
                ```python
                def general_purpose():
                    return "My purpose is to analyze the NVIDIA dataset and provide advanced analytics to help answer questions and gain insights from the data."
                ```

                User: Good morning

                Assistant Response:
                ```python
                def general_purpose():
                    return "Good morning. I'm ready to assist you with the dataset provided."
                ```

                User: Give a table to show the SKUs where the booked and shipped combined for SKUs are greater than the allocation quantities. The table should have SKU, booked plus shipped quantities and allocation quantity and greater than number

                Assistant Response:
                ```python
                def show_data(df):
                    # Filter data for Allocation, BOOKED and SHIPPED
                    filtered_df = df[df['Demand Type'].isin(['Allocation', 'BOOKED', 'SHIPPED'])]

                    # Group by SKU and calculate booked plus shipped quantities
                    booked_shipped_df = filtered_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Merge with allocation quantities
                    merged_df = pd.merge(booked_shipped_df, filtered_df[filtered_df['Demand Type'] == 'Allocation'][['SKU', 'Allocation Qty']], on='SKU', how='left')

                    # Calculate greater than number
                    merged_df['Greater Than Number'] = merged_df['Quantity'] - merged_df['Allocation Qty']

                    # Filter for SKUs where booked and shipped combined quantities are greater than allocation quantities
                    df_generated = merged_df[merged_df['Greater Than Number'] > 0][['SKU', 'Quantity', 'Allocation Qty', 'Greater Than Number']]

                    return df_generated
                ```

                User: For the third quarter of 2024 based on CRD dates, plot the top 20 booked and shipped quantities

                Assistant Response:
                ```python
                import pandas as pd
                import plotly.express as px

                def create_plot(df):
                    # Convert CRD Date to datetime
                    df['CRD Date'] = pd.to_datetime(df['CRD Date'])

                    # Filter data for BOOKED and SHIPPED in Q3 2024
                    booked_shipped_df = df[(df['Demand Type'].isin(['BOOKED', 'SHIPPED'])) & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)]

                    # Group by Cust Name and calculate booked and shipped quantities
                    booked_shipped_df = booked_shipped_df.groupby('Cust Name')['Quantity'].sum().reset_index()

                    # Sort by Quantity in descending order and get top 20
                    booked_shipped_df = booked_shipped_df.sort_values(by='Quantity', ascending=False).head(20)

                    # Create bar plot
                    fig = px.bar(booked_shipped_df, x='Cust Name', y='Quantity', title='Top 20 Customers by Booked and Shipped Quantities', color_discrete_sequence=['#76B900'])

                    return fig
                ```

                User: for BOARD customer, give the top 50 SKUs and plot them based on shippings

                Assistant Response:
                ```python
                import pandas as pd
                import plotly.express as px

                def create_plot(df):
                    # Filter data for BOARD customer and SHIPPED
                    shipped_df = df[(df['Cust Name'] == 'BOARD') & (df['Demand Type'] == 'SHIPPED')]

                    # Group by SKU and calculate shipped quantities
                    shipped_df = shipped_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Sort by Quantity in descending order and get top 50
                    shipped_df = shipped_df.sort_values(by='Quantity', ascending=False).head(50)

                    # Create bar plot
                    fig = px.bar(shipped_df, x='SKU', y='Quantity', title='Top 50 SKUs for BOARD Customer Based on Shippings', color_discrete_sequence=['#76B900'])

                    return fig
                ```

                User: For the fourth quarter of 2024, give the top 100 SKUs for allocation

                Assistant Response:
                ```python
                import pandas as pd

                def show_data(df):
                    # Convert Allocation Date to datetime
                    df['Allocation Date'] = pd.to_datetime(df['Allocation Date'])

                    # Filter data for Allocation in Q4 2024
                    allocation_df = df[(df['Demand Type'] == 'Allocation') & (df['Allocation Date'].dt.year == 2024) & (df['Allocation Date'].dt.quarter == 4)]

                    # Group by SKU and calculate allocation quantities
                    allocation_df = allocation_df.groupby('SKU')['Allocation Qty'].sum().reset_index()

                    # Sort by Allocation Qty in descending order and get top 100
                    df_generated = allocation_df.sort_values(by='Allocation Qty', ascending=False).head(100)

                    return df_generated
                ```

                User: Give me the SKUs with top 10 allocation

                Assistant Response:
                ```python
                import pandas as pd

                def show_data(df):
                    # Filter data for Allocation
                    allocation_df = df[df['Demand Type'] == 'Allocation']

                    # Group by SKU and calculate allocation quantities
                    allocation_df = allocation_df.groupby('SKU')['Allocation Qty'].sum().reset_index()

                    # Sort by Allocation Qty in descending order and get top 10
                    df_generated = allocation_df.sort_values(by='Allocation Qty', ascending=False).head(10).reset_index(drop=True)

                    return df_generated
                ```

                User: Give me the SKUs where the booked and shipped are greater than the supply for the second quarter of 2024 for CRD date

                Assistant Response:
                ```python
                import pandas as pd

                def show_data(df):
                    # Convert CRD Date to datetime
                    df['CRD Date'] = pd.to_datetime(df['CRD Date'])

                    # Filter data for Allocation, Supply, BOOKED and SHIPPED in Q2 2024
                    allocation_df = df[(df['Demand Type'] == 'Allocation') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 2)]
                    supply_df = df[(df['Demand Type'] == 'Supply') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 2)]
                    booked_shipped_df = df[(df['Demand Type'].isin(['BOOKED', 'SHIPPED'])) & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 2)]

                    # Group by SKU and calculate booked and shipped quantities
                    booked_shipped_df = booked_shipped_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Group by SKU and calculate supply quantities
                    supply_df = supply_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Merge with supply quantities
                    merged_df = pd.merge(booked_shipped_df, supply_df, on='SKU', how='outer')

                    # Rename Quantity column to Booked + Shipped
                    merged_df = merged_df.rename(columns={{'Quantity_x': 'Booked + Shipped', 'Quantity_y': 'Supply'}})

                    # Fill NaN values with 0
                    merged_df['Booked + Shipped'] = merged_df['Booked + Shipped'].fillna(0)
                    merged_df['Supply'] = merged_df['Supply'].fillna(0)

                    # Calculate greater than number
                    merged_df['Booked + Shipped > Supply'] = merged_df['Booked + Shipped'] - merged_df['Supply']

                    # Group by SKU and sum quantities
                    merged_df = merged_df.groupby('SKU')[['Booked + Shipped', 'Supply', 'Booked + Shipped > Supply']].sum().reset_index()

                    # Filter for SKUs where booked and shipped combined quantities are greater than supply quantities
                    df_generated = merged_df[merged_df['Booked + Shipped > Supply'] > 0][['SKU', 'Booked + Shipped', 'Supply', 'Booked + Shipped > Supply']].reset_index(drop=True)

                    return df_generated
                ```

                User: Give me a pie plot to show the top 20 SKUs where the booked and shipped is greater than supply for the third quarter of 2024

                Assistant Response:
                ```python
                import pandas as pd
                import plotly.express as px

                def create_plot(df):
                    # Convert CRD Date to datetime
                    df['CRD Date'] = pd.to_datetime(df['CRD Date'])

                    # Filter data for Allocation, Supply, BOOKED and SHIPPED in Q3 2024
                    allocation_df = df[(df['Demand Type'] == 'Allocation') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)]
                    supply_df = df[(df['Demand Type'] == 'Supply') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)]
                    booked_shipped_df = df[(df['Demand Type'].isin(['BOOKED', 'SHIPPED'])) & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)]

                    # Merge with supply quantities
                    merged_df = pd.merge(booked_shipped_df.groupby('SKU')['Quantity'].sum().reset_index(), 
                                        supply_df.groupby('SKU')['Quantity'].sum().reset_index(), 
                                        on='SKU', how='outer')

                    # Rename Quantity column to Booked + Shipped
                    merged_df = merged_df.rename(columns={{'Quantity_x': 'Booked + Shipped', 'Quantity_y': 'Supply'}})

                    # Fill NaN values with 0
                    merged_df['Booked + Shipped'] = merged_df['Booked + Shipped'].fillna(0)
                    merged_df['Supply'] = merged_df['Supply'].fillna(0)

                    # Calculate greater than number
                    merged_df['Booked + Shipped > Supply'] = merged_df['Booked + Shipped'] - merged_df['Supply']

                    # Filter for SKUs where booked and shipped combined quantities are greater than supply quantities
                    merged_df = merged_df[merged_df['Booked + Shipped > Supply'] > 0]

                    # Sort by Booked + Shipped > Supply in descending order and get top 20
                    merged_df = merged_df.sort_values(by='Booked + Shipped > Supply', ascending=False).head(20)

                    # Create pie plot
                    fig = px.pie(merged_df, names='SKU', values='Booked + Shipped > Supply', title='Top 20 SKUs where Booked and Shipped is greater than Supply for Q3 2024', color_discrete_sequence=['#76B900'])

                    return fig
                ```

                User: How are you doing?

                Assistant Response:
                ```python
                def general_purpose():
                    return "I'm doing well, thank you. My purpose is to analyze the NVIDIA dataset and provide advanced analytics to help answer questions and gain insights from the data."
                ```

                User: Can you help in forecasting?

                Assistant Response:
                ```python
                def general_purpose():
                    return "Yes, I can help with forecasting. Please note that forecasting is still a beta feature and I am continuously improving it. I can use xgboost regressor to build a forecast using lagged features."
                ```

                User: What is your purpose?

                Assistant Response:
                ```python
                def general_purpose():
                    return "My purpose is to analyze the NVIDIA dataset and provide advanced analytics to help answer questions and gain insights from the data."
                ```

                User: Can you delete datasets?

                Assistant Response:
                ```python
                def general_purpose():
                    return "No, I am not capable of deleting datasets. My purpose is to analyze the NVIDIA dataset and provide advanced analytics to help answer questions and gain insights from the data."
                ```

                User: For each of the demand type, how many SKUs are there?

                Assistant Response:
                ```python
                def data_exploration(df):
                    # Get the number of unique SKUs for each demand type
                    num_skus = df.groupby('Demand Type')['SKU'].nunique().reset_index()

                    return f"The number of SKUs for each demand type is: \n{{num_skus}}"
                ```

                User: For the third quarter of 2024, give the total for each of the demand types

                Assistant Response:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Convert Allocation Date and CRD Date to datetime
                    df['Allocation Date'] = pd.to_datetime(df['Allocation Date'])
                    df['CRD Date'] = pd.to_datetime(df['CRD Date'])

                    # Filter data for Q3 2024
                    q3_2024_df = df[(df['Allocation Date'].dt.year == 2024) & (df['Allocation Date'].dt.quarter == 3)]

                    # Get the total quantities for each demand type
                    total_quantities = q3_2024_df.groupby('Demand Type')

                    # For allocation, look for allocation quantity
                    allocation_quantity = total_quantities.get_group('Allocation')['Allocation Qty'].sum()

                    # For other demand types, look for quantity
                    other_quantities = df[(df['Demand Type'] != 'Allocation') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)].groupby('Demand Type')['Quantity'].sum().reset_index()

                    # Rename columns
                    other_quantities.columns = ['Demand Type', 'Quantity']

                    # Add allocation quantity to other quantities
                    other_quantities.loc[len(other_quantities.index)] = ['Allocation', allocation_quantity]

                    return f"The total quantities for each demand type in Q3 2024 are: \n{{other_quantities}}"
                ```

                User: Give me the top 20 SKUs where the demand is greater than the allocation

                Assistant Response:
                ```python
                import pandas as pd
                import plotly.express as px

                def create_plot(df):
                    # Filter data for Allocation, BOOKED and SHIPPED
                    allocation_df = df[df['Demand Type'] == 'Allocation']
                    booked_shipped_df = df[df['Demand Type'].isin(['BOOKED', 'SHIPPED'])]

                    # Group by SKU and calculate allocation quantities
                    allocation_df = allocation_df.groupby('SKU')['Allocation Qty'].sum().reset_index()

                    # Group by SKU and calculate booked and shipped quantities
                    booked_shipped_df = booked_shipped_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Merge with booked and shipped quantities
                    merged_df = pd.merge(allocation_df, booked_shipped_df, on='SKU', how='outer')

                    # Rename Quantity column to Booked + Shipped
                    merged_df = merged_df.rename(columns={{'Quantity': 'Booked + Shipped'}})

                    # Fill NaN values with 0
                    merged_df['Allocation Qty'] = merged_df['Allocation Qty'].fillna(0)
                    merged_df['Booked + Shipped'] = merged_df['Booked + Shipped'].fillna(0)

                    # Calculate greater than number
                    merged_df['Booked + Shipped > Allocation Qty'] = merged_df['Booked + Shipped'] - merged_df['Allocation Qty']

                    # Filter for SKUs where booked and shipped combined quantities are greater than allocation quantities
                    merged_df = merged_df[merged_df['Booked + Shipped > Allocation Qty'] > 0]

                    # Sort by Booked + Shipped > Allocation Qty in descending order and get top 20
                    merged_df = merged_df.sort_values(by='Booked + Shipped > Allocation Qty', ascending=False).head(20)

                    # Create bar plot
                    fig = px.bar(merged_df, x='SKU', y='Booked + Shipped > Allocation Qty', title='Top 20 SKUs where Demand is greater than Allocation', color_discrete_sequence=['#76B900'])

                    return fig
                ```

                User: Give the top 50 SKUs in WS business unit where the demand is greater tahn the supply for 3rd quarter of 2024

                Assistant: 
                ```python
                import pandas as pd
                import plotly.express as px

                def create_plot(df):
                    # Convert CRD Date to datetime
                    df['CRD Date'] = pd.to_datetime(df['CRD Date'])

                    # Filter data for Supply, BOOKED and SHIPPED in Q3 2024 for WS business unit
                    supply_df = df[(df['Demand Type'] == 'Supply') & (df['Business Unit'] == 'WS') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)]
                    booked_shipped_df = df[(df['Demand Type'].isin(['BOOKED', 'SHIPPED'])) & (df['Business Unit'] == 'WS') & (df['CRD Date'].dt.year == 2024) & (df['CRD Date'].dt.quarter == 3)]

                    # Group by SKU and calculate supply quantities
                    supply_df = supply_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Group by SKU and calculate booked and shipped quantities
                    booked_shipped_df = booked_shipped_df.groupby('SKU')['Quantity'].sum().reset_index()

                    # Merge with supply quantities
                    merged_df = pd.merge(booked_shipped_df, supply_df, on='SKU', how='outer')

                    # Rename Quantity column to Booked + Shipped
                    merged_df = merged_df.rename(columns={{'Quantity_x': 'Booked + Shipped', 'Quantity_y': 'Supply'}})

                    # Fill NaN values with 0
                    merged_df['Booked + Shipped'] = merged_df['Booked + Shipped'].fillna(0)
                    merged_df['Supply'] = merged_df['Supply'].fillna(0)

                    # Calculate greater than number
                    merged_df['Booked + Shipped > Supply'] = merged_df['Booked + Shipped'] - merged_df['Supply']

                    # Filter for SKUs where booked and shipped combined quantities are greater than supply quantities
                    merged_df = merged_df[merged_df['Booked + Shipped > Supply'] > 0]

                    # Sort by Booked + Shipped > Supply in descending order and get top 50
                    merged_df = merged_df.sort_values(by='Booked + Shipped > Supply', ascending=False).head(50)

                    # Create bar plot
                    fig = px.bar(merged_df, x='SKU', y='Booked + Shipped > Supply', title='Top 50 SKUs where Demand is greater than Supply for WS Business Unit', color_discrete_sequence=['#76B900'])

                    return fig
                ```

                User: What is the demand, supply and allocation for SKU A?

                Assistant:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Filter data for A
                    A_df = df[df['SKU'] == 'A']

                    # Filter data for Allocation
                    allocation_df = A_df[A_df['Demand Type'] == 'Allocation']

                    # Filter data for Supply
                    supply_df = A_df[A_df['Demand Type'] == 'Supply']

                    # Filter data for BOOKED and SHIPPED
                    booked_shipped_df = A_df[A_df['Demand Type'].isin(['BOOKED', 'SHIPPED'])]

                    # Get the allocation quantity
                    allocation_quantity = allocation_df['Allocation Qty'].sum()

                    # Get the supply quantity
                    supply_quantity = supply_df['Quantity'].sum()

                    # Get the demand quantity
                    demand_quantity = booked_shipped_df['Quantity'].sum()

                    return f"The demand for A is {{demand_quantity}}, the supply is {{supply_quantity}}, and the allocation is {{allocation_quantity}}"
                ```

                User: What is the demand for SKU 123-1512?

                Assistant:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Filter data for BOOKED and SHIPPED
                    booked_shipped_df = df[df['Demand Type'].isin(['BOOKED', 'SHIPPED'])]

                    # Filter data for SKU 123-1512
                    sku_df = booked_shipped_df[booked_shipped_df['SKU'] == '123-1512']

                    # Check if SKU exists in the dataset
                    if sku_df.empty:
                        return f"SKU 123-1512 does not exist in the dataset based on the selected demand type."
                    else:
                        # Get the demand for SKU 123-1512
                        demand = sku_df['Quantity'].sum()
                        return f"The demand for SKU 123-1512 is {{demand}}"
                ```

                User: What is the total supply available?

                Assistant:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Filter data for Supply
                    supply_df = df[df['Demand Type'] == 'Supply']

                    # Get the total supply
                    total_supply = supply_df['Quantity'].sum()

                    # Check if total supply is 0
                    if total_supply == 0:
                        return "The total supply available is 0."
                    else:
                        return f"The total supply available is {{total_supply}}"
                ```

                User: What is the total supply available?

                Assistant:
                ```python
                import pandas as pd

                def data_exploration(df):
                    # Filter data for Supply
                    supply_df = df[df['Demand Type'] == 'Supply']

                    # Check if supply data is empty
                    if supply_df.empty:
                        return "There is no supply data available. Please select 'Supply' in the demand type section to perform analysis of the supply."
                    else:
                        # Get the total supply
                        total_supply = supply_df['Quantity'].sum()

                        # Check if total supply is 0
                        if total_supply == 0:
                            return "The total supply available is 0."
                        else:
                            return f"The total supply available is {{total_supply}}"
                ```

                ALWAYS ALWAYS ALWAYS use the code by Assistant as shown below if the question is the following by the user. 

                User: For the top 10 SKUs based on demand for MB business unit, show the variation in demand

                Assistant:
                ```python
                import pandas as pd
                import plotly.express as px

                def create_plot(df):
                    # Filter data for MB business unit and BOOKED and SHIPPED demand types
                    df_filtered = df[(df['Business Unit'] == 'MB') & (df['Demand Type'].isin(['BOOKED', 'SHIPPED']))]

                    # Calculate top 10 SKUs by total demand
                    top_10_skus = df_filtered.groupby('SKU')['Quantity'].sum().nlargest(10).index
                    df_top_10 = df_filtered[df_filtered['SKU'].isin(top_10_skus)]

                    # Calculate IQR and remove extreme outliers
                    Q1 = df_top_10['Quantity'].quantile(0.25)
                    Q3 = df_top_10['Quantity'].quantile(0.75)
                    IQR = Q3 - Q1

                    # Exclude points strictly outside the IQR range
                    df_no_outliers = df_top_10[(df_top_10['Quantity'] >= (Q1 - 1.5 * IQR)) & (df_top_10['Quantity'] <= (Q3 + 1.5 * IQR))]

                    # Create boxplot for top 10 SKUs without showing any additional points
                    fig = px.box(
                        df_no_outliers, 
                        x='SKU', 
                        y='Quantity', 
                        title='Variation in Demand for Top 10 SKUs in MB Business Unit (outliers removed)',
                        points=False,  # Disable additional points outside IQR
                        color_discrete_sequence=['#76B900']
                    )

                    fig.update_layout(
                        xaxis_title='SKU',
                        yaxis_title='Quantity',
                        showlegend=False
                    )

                    return fig
                ```
            """

        if "messages" not in st.session_state:

            st.session_state.messages = ["placeholder"]

        st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}

        with tab_selected[1]:

            if st.button("📝 Click for new chat", help="Click for new chat"):
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


        # with tab_selected[1]:
        #     if st.button("📝 Click for new chat", help="Click for new chat"):
        #         st.session_state.messages = []
        #         st.session_state.messages_display = []
        #         st.session_state.debugger_messages = []
        #         st.session_state.messages = ["placeholder"]
        #         st.session_state.messages[0] = {"role": "system", "content": system_message_file_uploaded}
        #         st.session_state.flag_new_chat = 1
        #         now = datetime.now()
        #         st.session_state.new_formatted_date_time = now.strftime("%B %d, %Y %I:%M:%S %p")
        #         st.session_state.conversations[f"{st.session_state.new_formatted_date_time}"] = []
        #         st.session_state.conversation_selected = st.session_state.new_formatted_date_time
        #         # st.write(st.session_state.new_formatted_date_time)

        #     for selected_conversation in list(st.session_state.conversations.keys()):
        #         if st.button(selected_conversation):
        #             st.session_state.conversation_selected = selected_conversation
        #             st.session_state.button_clicked_flag = True
        #             # st.session_state.messages_display = st.session_state.conversations[selected_conversation]

        # if st.session_state.flag_new_chat == 0:
        #     st.session_state.conversation_selected = st.session_state.old_formatted_date_time
        #     st.session_state.conversations[f'{st.session_state.conversation_selected}'] = st.session_state.messages_display
        
        # if st.session_state.flag_new_chat == 1:
        #     st.session_state.conversation_selected = st.session_state.new_formatted_date_time
        #     # st.write(st.session_state.conversation_selected)

        #     # st.write(st.session_state.messages)
        #     st.session_state.conversations[f'{st.session_state.conversation_selected}'] = st.session_state.messages_display


        i = 0
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
            if message['role'] == 'show_data' or message['role'] == 'show_diff_data':
                st.dataframe(message['dataframe'], width=column_width)
            if message['role'] == 'forecast plot':
                st.plotly_chart(message['figure'])


        if flag == 1:

            if prompt := st.chat_input("Write your lines here..."):

                # additional_message = """
                # INSTRUCTIONS 
                # - If the query is outside the scope of the dataset, reply generally similar to "I'm ready to assist you with the dataset provided" and similar responses and NEVER anything else.
                # - If it is a general purpose query, let the user only know your purpose briefly.
                # - Import the necessary libraries which are needed for the task.
                # - Only use one or more of these functions and do not give extra code of example uses or extra information. The output 
                # should be only the functions.

                # a) create_plot(df)
                # b) data_exploration(df)
                # c) show_data(df)
                
                # a) create_plot(df) - Use this function when asked to plot. Returns 'fig' which is a figure. Use only plotly for visualizations
                # b) data_exploration(df) - Use this function when you want to answer the queries from the user about the dataset. The output should be a string with insights based on the user query.
                # c) show_data(df) - You can also use 'show_data' function to return a single dataframe called 'df_generated'. The output should be one argument. The input should be single argument 'df' as a dataframe.

                # We are using streamlit and the following are the colors used.

                # Primary color - #76B900
                # Background color - #1C1C1C
                # Text color - #F1F1F1
                # Secondary background color - #2E2E2E

                # Be sure to use only NVIDIA colors when plotting which go well with the streamlit colors.
                # """

                #                 - If the query is outside the scope of the dataset, reply generally similar to "I'm ready to assist you with the dataset provided" and similar responses and NEVER anything else.

                additional_message = f"""
                INSTRUCTIONS 
                - If it is a general purpose query, let the user only know your purpose briefly.
                - Import the necessary libraries which are needed for the task.
                - Only use one or more of these functions and do not write code outside of the functions. The output 
                should be only the functions.
                - Try to give as accurate and executable code as possible without syntax errors.

                a) create_plot(df)
                b) data_exploration(df)
                c) show_data(df)
                d) generate_forecasts(df)
                e) general_purpose()
                
                a) create_plot(df) - Use this function when asked to plot. Returns 'fig' which is a figure. Use only plotly for visualizations. The output could either be plots of the insights from changes as a single return output. When using this, ensure to give proper titles and also necessary labels for plots.
                b) data_exploration(df) - Use this function when you want to answer the queries from the user about the dataset. The output should be a string with insights based on the user query.
                c) show_data(df) - You can also use 'show_data' function to return a single dataframe called 'df_generated'. The output should be one argument. The input should be single argument 'df' as a dataframe. 
                d) generate_forecasts(df) - Use this function when asked to forecast. Can either return 'fig' as the output or a table. Use only plotly for visualizations. The output could either be plots of the insights from changes as a single return output. When using this, ensure to give proper titles and also necessary labels for plots.
                Use xgboost regressor to build a forecast using lagged features.
                e) general_purpose() - Use this function to answer general purpose questions very briefly but reminding about your purpose of analyzing the data and providing advanced analytics for Nvidia dataset. Return a string.

                The following are the available demand types from the data selected by user and the data is available only fro these categories: {categorical_columns_dict['Demand Type']}
                The following are a list of all the demand types available though the user selected some or all of them: ['BOOKED', 'SHIPPED', 'Supply', 'Allocation', 'Locked Allocation', 'JFF', 'ZSCH', 'BUF', 'RSF', 'PULLED']
                Be sure to always filter the data based on these demand types. 
                If the question is about allocation or locked allocation and when the demand type present in the data has 'Allocation' or 'Locked Allocation', 
                ensure to look for 'Allocation Qty' for quantities and dates as 'Allocation Date'.
                Be sure to use only NVIDIA colors #76B900 when plotting.
                """

                # st.write(additional_message)

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

                    if 'create_plot' in file_read:
                        try:
                            exec(file_read, exec_namespace)
                            fig_returned = exec_namespace['create_plot'](dfs)
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
                                        fig_returned = exec_namespace['create_plot'](dfs)
                                        error_occurred = False
                                    except:
                                        error_count = error_count + 1
                                        if error_count > debug_attempts:
                                            error_occurred = False

                            st.plotly_chart(fig_returned)
                            st.session_state.messages_display.append({'role': 'plot', 'figure': fig_returned})  

                    if 'show_data' in file_read:
                        try:
                            exec(file_read, exec_namespace)
                            df_generated = exec_namespace['show_data'](dfs)
                            if color_coded_data == "Yes":
                                styled_df = df_generated.style.applymap(color_code)
                                st.dataframe(styled_df, width=column_width)
                                st.session_state.messages_display.append({'role': 'show_data', 'dataframe': styled_df})   
                            else:
                                df_generated.reset_index(drop=True, inplace=True)
                                st.dataframe(df_generated, width=column_width)
                                st.session_state.messages_display.append({'role': 'show_data', 'dataframe': df_generated})   
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
                                        df_generated= exec_namespace['show_data'](dfs)
                                        error_occurred = False
                                    except:
                                        error_count = error_count + 1
                                        if error_count > debug_attempts:
                                            error_occurred = False
                        
                            if color_coded_data == "Yes":
                                styled_df = df_generated.style.applymap(color_code)
                                st.dataframe(styled_df, width=column_width)
                                st.session_state.messages_display.append({'role': 'show_data', 'dataframe': styled_df})   
                            else:
                                df_generated.reset_index(drop=True, inplace=True)
                                st.dataframe(df_generated, width=column_width)
                                st.session_state.messages_display.append({'role': 'show_data', 'dataframe': df_generated})   

                            # st.dataframe(df_generated, width=column_width)
                            # st.session_state.messages_display.append({'role': 'show_data', 'dataframe': df_generated})   
                            
                    if 'data_exploration' in file_read:
                        try:
                            exec(file_read, exec_namespace)
                            message_returned = exec_namespace['data_exploration'](dfs)
                            # st.write(message_returned)

                            if not private_nvcf_endpoints:

                                client = OpenAI(
                                    base_url=base_url,
                                    api_key=api_key
                                )

                                stream = client.chat.completions.create(
                                    model=st.session_state["llm_model"],
                                    messages=[
                                        {'role': 'user', 'content': message_returned + ". Rewrite this exactly the same except make it legible and only the statement should appear and not additional information. Note that this iS NVIDIA dataset for supply chain."}
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
                                        message_returned = exec_namespace['data_exploration'](dfs)
                                        error_occurred = False
                                    except:
                                        error_count = error_count + 1
                                        if error_count > debug_attempts:
                                            break
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

                    if 'generate_forecasts' in file_read:
                        try:
                            exec(file_read, exec_namespace)
                            fig_returned = exec_namespace['generate_forecasts'](dfs)
                            st.plotly_chart(fig_returned)
                            st.session_state.messages_display.append({'role': 'forecast plot', 'figure': fig_returned}) 
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
                                        fig_returned = exec_namespace['generate_forecasts'](dfs)
                                        error_occurred = False
                                    except:
                                        error_count = error_count + 1
                                        if error_count > debug_attempts:
                                            error_occurred = False

                            st.plotly_chart(fig_returned)
                            st.session_state.messages_display.append({'role': 'forecast plot', 'figure': fig_returned})

                    if 'general_purpose' in file_read:

                        exec(file_read, exec_namespace)
                        message_returned = exec_namespace['general_purpose']()

                        if not private_nvcf_endpoints:
                            client = OpenAI(
                                base_url=base_url,
                                api_key=api_key
                            )

                            stream = client.chat.completions.create(
                                model=st.session_state["llm_model"],
                                messages=[
                                    {'role': 'user', 'content': f"The following is the input message given by user: {prompt}" + ".\n" + f"The following is the reply by the large language model: {message_returned}" + ". Rewrite only the reply for this by the large language model while keeping the context of the question and tell this as if you were speaking to demand planner from Nvidia. If the message is outside the scope of the NVIDIA dataset, be sure to let them know your purpose and not answer question about the user input and say sorry but be sure to always be polite and very respectful. Do not use quotes."}
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

# Conditional rendering of pages based on login status
if st.session_state['logged_in']:
    if st.session_state['task_submitted']:
        content_page()
    else:
        task_page()
        # content_page()  # Show the second page if logged in
else:
    login_page()    # Show the login page if not logged in


   



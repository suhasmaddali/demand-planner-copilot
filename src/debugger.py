from openai import OpenAI
import streamlit as st
import os
from langchain_nvidia_ai_endpoints import register_model, Model, ChatNVIDIA

# base_url = st.secrets['USER CREDENTIALS']['BASE_URL']
# api_key = st.secrets['USER CREDENTIALS']['API_KEY']

# # Getting access to NVCF credentials for inference
# NVCF_CHAT_FUNCTION_ID = st.secrets['NVCF CREDENTIALS']['NVCF_CHAT_FUNCTION_ID']
# NVCF_URL = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{NVCF_CHAT_FUNCTION_ID}"
# NVCF_API_KEY = st.secrets['NVCF CREDENTIALS']['NVCF_API_KEY']
# MODEL = "meta/llama-3.1-8b-instruct"
# os.environ['NVIDIA_API_KEY'] = NVCF_API_KEY

base_url = os.getenv('USER_CREDENTIALS_BASE_URL')
api_key = os.getenv('USER_CREDENTIALS_API_KEY')

NVCF_CHAT_FUNCTION_ID = os.getenv('NVCF_CHAT_FUNCTION_ID')
NVCF_URL = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{NVCF_CHAT_FUNCTION_ID}"
NVCF_API_KEY = os.getenv('NVCF_API_KEY')
MODEL = "meta/llama-3.1-8b-instruct"
os.environ['NVIDIA_API_KEY'] = NVCF_API_KEY

class Debugger:
    def __init__(self, llm_model, kwargs, private_nvcf_endpoints: bool):

        self.enable_private_endpoints = private_nvcf_endpoints
        if self.enable_private_endpoints:
            register_model(Model(id=MODEL, model_type="chat", client="ChatNVIDIA", endpoint=NVCF_URL))

            self.client = ChatNVIDIA(
                model=MODEL,
                temperature=0.0,
                max_tokens=4096
            )
        else:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )

        self.llm_model = llm_model
        self.categorical_columns_dict = kwargs['categorical_columns_dict']
        self.numerical_columns_dict = kwargs['numerical_columns_dict']
        self.datetime_columns = kwargs['datetime_columns']

        # self.categorical_columns_dict2 = kwargs['categorical_columns_dict2']
        # self.numerical_columns_dict2 = kwargs['numerical_columns_dict2']
        # self.datetime_columns2 = kwargs['datetime_columns2']

        # self.initial_date_string = kwargs['initial_date_string']
        # self.final_date_string = kwargs['final_date_string']

        # self.system_message_file_uploaded = f"""            
        #     The following in double backticks (``) is provided for you to get context about the dataset.
        #     Do not use this information for plotting but only to help you get understanding of the dataset.
        #     ``
        #     The following are the categorical columns and the corresponding unique categories per column.

        #     <{self.categorical_columns_dict}>

        #     The following are the numerical columns present in the data.
        
        #     <{self.numerical_columns_dict}>

        #     The following are the datetime columns present in the data.

        #     <{self.datetime_columns}>

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
            
        #     Your task is to give the correct code based on the error code which is given by the user. 

        #     INSTRUCTIONS
        #     - ALWAYS give ONLY the corrected code and nothing else
        #     - USE SINGLE CODE BLOCK with a solution 
        #     - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK
        #     - The code block must start and end with ```python and then end with ```
        # """


        self.system_message_file_uploaded = f"""            
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
            
            Your task is to give the correct code based on the error code which is given by the user. 

            INSTRUCTIONS
            - ALWAYS give ONLY the corrected code and nothing else
            - USE SINGLE CODE BLOCK with a solution 
            - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK
            - The code block must start and end with ```python and then end with ```
        """

    def chat(self, messages):

        # output_stream = self.client.chat.completions.create(
        #     model=self.llm_model,
        #     messages=[
        #         {"role": "system", "content": self.system_message_file_uploaded},
        #         messages
        #     ],
        #     stream=stream,
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )

        if self.enable_private_endpoints:
            response_list = []
            botmsg = st.empty()
            for chunk in self.client.stream(
                [
                    {"role": "system", "content": self.system_message_file_uploaded},
                    messages
                ]
            ):
                response_list.append(chunk.content)
                result = "".join(response_list)
                botmsg.write(result + "▌")
            if result:
                botmsg.write(result) 
        else:
            result = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_message_file_uploaded},
                    messages
                ],
                stream=True,
                temperature=0,
                max_tokens=4096
            )



        return result



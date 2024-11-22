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


class Recommender:
    def __init__(self, llm_model, private_nvcf_endpoints: bool):

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

        self.system_message_file_uploaded = f"""                        
            Your task is to give a list of prompts based on all the conversation history given happening till now. 

            INSTRUCTIONS
            - ALWAYS begin the response with the list of prompts which user can ask based on the conversation history till now
            - ONLY include the prompts and nothing else
            - Give the prompts based on how many prompts do they need or number of suggestions which they need
            - After every suggested prompt, ALWAYS give \n to indicate the end of one suggested prompt
            - ALWAYS give the suggested prompts based on context which will be in [context]

            Below are some examples of how you should respond based on the complete context which is given to you. Note that context can be anything 
            and you should 

            User: 
            [context]

            Assistant Response: 
            What are the top 10 SKUs where the demand is greater than allocation\n
            What are the top 20 SKUs with highest demand\n
            Give a plot to show the top 10 SKUs with highest bookings\n
            Give a plot to show the top 20 SKUs with highest bookings\n
            Give a plot to show the top 50 SKUs with highest bookings\n

            User: 
            [context]

            Assistant Response:
            What is the total supply for the SKU 115-1136-000\n
            Give a plot to show the total demand for all the products combined\n
            Give a plot to show the top 10 SKUs based on supply

            User: 
            [context]

            Assistant Response:
            What is the total BUF for the SKU 115-1136-000\n
            Give a plot to show the total demand for all the products combined\n
            Give a plot to show the top 10 SKUs based on supply\n
            Give a table to highlight bookings vs allocation for SKUs
        """

    def chat(self, messages):

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
                stream=False,
                temperature=0,
                max_tokens=4096
            )

        return result



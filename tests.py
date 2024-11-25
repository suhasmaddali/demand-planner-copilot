# import streamlit as st
# import random
# import time


# # Streamed response emulator
# def response_generator():
#     response = random.choice(
#         [
#             "Hello there! How can I assist you today?",
#             "Hi, human! Is there anything I can help you with?",
#             "Do you need help?",
#         ]
#     )
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)


# st.title("Simple chat")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What is up?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         response = st.write_stream(response_generator())
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})


# st.sidebar.title("Possible prompts")

# total_suggestions = st.sidebar.number_input("Number of suggestions:", value=5)

# prompts = [
#     "What are the benefits of yoga?",
#     "How does meditation improve mental health?",
#     "What are the best exercises for weight loss?",
#     "How can I improve my time management skills?",
#     "What are the advantages of eating a balanced diet?",
#     "What are effective ways to reduce stress?",
#     "How to improve focus and concentration?",
#     "What are the benefits of daily journaling?",
#     "How does drinking water impact health?",
#     "What are the best practices for better sleep?",
#     "How can I stay motivated to achieve my goals?",
#     "What are the top tips for improving communication skills?",
#     "How does regular exercise impact mental health?",
#     "What are the benefits of practicing gratitude daily?",
#     "How can I build a consistent morning routine?",
#     "What are the advantages of learning a new skill?",
#     "How does spending time in nature improve well-being?",
#     "What are the best ways to develop a growth mindset?",
#     "How can I reduce screen time effectively?",
#     "What are the benefits of volunteering and helping others?"
# ]

# prompts = prompts[:total_suggestions]

# random.shuffle(prompts)

# # Create buttons dynamically
# for prompt in prompts:
#     if st.sidebar.button(prompt):
#         st.session_state.button_clicked = True
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#             response = st.write_stream(response_generator())  # Replace with your response logic
#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})



# import streamlit as st
# import pandas as pd
# from openai import OpenAI
# base_url = st.secrets['USER CREDENTIALS']['BASE_URL']
# api_key = st.secrets['USER CREDENTIALS']['API_KEY']
# import time

# def stream_responses(input_stream):

#     response_list = []
#     botmsg = st.empty()
#     for chunk in input_stream:
#         text = chunk.choices[0].delta.content
#         if text:
#             response_list.append(text)
#             result = "".join(response_list).strip()
#             botmsg.write(result + "▌")
#             time.sleep(0.05)
#     if result:
#         botmsg.write(result)  
#     return result

# @st.dialog("Cast your vote", width="large")
# def vote(item):

#     if 'messages_display' not in st.session_state:
#         st.session_state.messages_display = []

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     if 'debugger_messages' not in st.session_state:
#         st.session_state.debugger_messages = []

#     st.write(f"Why is {item} your favorite?")
#     # Generate a larger dataset
#     data = {
#         "ID": list(range(1, 21)),  # 20 unique IDs
#         "Name": [f"Person {i}" for i in range(1, 21)],  # Names from Person 1 to Person 20
#         "Age": [25 + (i % 10) for i in range(20)],  # Ages between 25 and 34
#         "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"] * 4,  # Repeat cities
#         "Salary": [50000 + (i * 1000) for i in range(20)]  # Salaries increasing by $1000
#     }

#     # Convert the dictionary to a pandas DataFrame
#     df = pd.DataFrame(data)

#     st.dataframe(df)

#     # Display chat history in a scrollable container
#     chat_container = st.container()
#     with chat_container:
#         for message in st.session_state["messages"]:
#             st.write(message)

#     # Add a chat input at the bottom of the dialog
#     user_input = st.chat_input("Enter your message")

#     if user_input:
#         # Add the user's input to the chat history
#         st.session_state["messages"].append(f"You: {user_input}")

#         # Add a sample bot response (for demonstration)
#         st.session_state["messages"].append(f"Bot: Echoing '{user_input}'")

        # Scroll the container to the bottom (Streamlit handles this automatically)
        # st.rerun()

    # if 'messages_display' not in st.session_state:
    #     st.session_state.messages_display = []

    # if 'messages' not in st.session_state:
    #     st.session_state.messages = []

    # if 'debugger_messages' not in st.session_state:
    #     st.session_state.debugger_messages = []

    # # for message in st.session_state.messages_display:
    # #     if message['role'] == 'user':
    # #         with st.chat_message(message['role'], avatar="🔍"):
    # #             st.markdown(message['content'])
    # #     if message['role'] == 'assistant':
    # #         with st.status("📟 *Generating the code*..."):
    # #             with st.chat_message(message['role'], avatar="🤖"):
    # #                 st.markdown(message['content'])
    # #     if message['role'] == 'plot':
    # #         st.plotly_chart(message['figure'])
    # #     if message['role'] == 'adhoc':
    # #         st.write(message['message from adhoc query'])
    # #     if message['role'] == 'show_diff_data':
    # #         st.dataframe(message['dataframe'])

    # if prompt := st.chat_input("Write your lines here..."):

    #     client = OpenAI(
    #             base_url=base_url,
    #             api_key=api_key
    #         )
    #     st.session_state["llm_model"] = "meta/llama-3.1-70b-instruct"
    #     st.session_state["debugger"] = "meta/llama-3.1-70b-instruct"


    #     if 'clear_chat_history_button' not in st.session_state:
    #         st.session_state.clear_chat_history_button = False
    #     additional_message = f"""
    #     INSTRUCTIONS 
    #     - If it is a general purpose query, NEVER give a reply but reply saying that you are ready to assist with the dataset.
    #     - Import the necessary libraries which are needed for the task.
    #     - Only use one or more of these functions and do not write code outside of the functions. The output 
    #     should be only the functions.
    #     """

    #     # which is an ExcelFile format along with 'sheet_name'. The 'sheet_name' should be a keyword argument.
    #     enhanced_prompt = prompt + additional_message
    #     st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
    #     st.session_state.messages_display.append({'role': 'user', 'content': prompt})
    #     with st.chat_message("user", avatar="🔍"):
    #         st.markdown(prompt)

    #     # st.write(st.session_state.messages)

        
    #     # st.write([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
        

    #     status_placeholder = st.empty()

    #     # st.write(st.session_state.messages)

    #     with status_placeholder.status("📟 *Generating the code*..."):
    #         with st.chat_message("assistant", avatar="🤖"):
    #             stream = client.chat.completions.create(
    #                 model=st.session_state["llm_model"],
    #                 messages=[
    #                     {"role": m["role"], "content": m["content"]}
    #                     for m in st.session_state.messages
    #                 ],
    #                 stream=True,
    #                 temperature=0.0,
    #                 max_tokens=4096
    #             )

    #             result = stream_responses(input_stream=stream)   

    #     st.session_state.messages.append({"role": "assistant", "content": result})
    #     st.session_state.messages_display.append({'role': 'assistant', 'content': result})
    
    # reason = st.text_input("Because...")
    # if st.button("Submit"):
    #     st.session_state.vote = {"item": item, "reason": reason}
    #     st.rerun()

# if "vote" not in st.session_state:
#     st.write("Vote for your favorite")
#     if st.button("A"):
#         vote("A")
#     if st.button("B"):
#         vote("B")
# else:
#     f"You voted for {st.session_state.vote['item']} because {st.session_state.vote['reason']}"



# import streamlit as st

# # st.context.headers["Suhas"] = "Maddali"

# st.context.headers

# st.write(st.context.cookies.keys())

# st.context.cookies.



# import streamlit as st
# import asyncio

# # Simulate an asynchronous API call
# async def async_task(duration):
#     await asyncio.sleep(duration)  # Simulate I/O-bound task
#     return f"Task completed in {duration} seconds!"

# # Main Streamlit App
# st.title("Async Streamlit Application Example")

# # User input for task duration
# duration = st.number_input("Enter duration for async task (seconds):", min_value=1, max_value=10, value=3)

# # Button to start the async task
# if st.button("Run Async Task"):
#     with st.spinner("Running async task..."):
#         # Run the async task
#         result = asyncio.run(async_task(duration))
#         st.success(result)

# st.write("This is a basic example of integrating async tasks into Streamlit.")





# import streamlit as st

# # Create a placeholder for output
# output_placeholder = st.empty()

# # Create tabs
# tab1, tab2 = st.tabs(["Configuration", "Another Tab"])

# # Configuration Tab
# with tab1:
#     st.subheader("Configuration Tab")
#     if st.button("Generate Output"):
#         # Update the placeholder directly
#         with output_placeholder:
#             st.write("### Output on the Main Screen:")
#             st.write("This is the output generated in the Configuration tab!")

# # Another Tab
# with tab2:
#     st.subheader("Another Tab")
#     st.write("Content for another tab.")




# import streamlit as st

# @st.dialog("Cast your vote")
# def vote(item):
#     st.write(f"Why is {item} your favorite?")
#     reason = st.text_input("Because...")
#     if st.button("Submit"):
#         st.session_state.vote = {"item": item, "reason": reason}
#         st.rerun()

# if "vote" not in st.session_state:
#     st.write("Vote for your favorite")
#     if st.button("A"):
#         vote("A")
#     if st.button("B"):
#         vote("B")
# else:
#     f"You voted for {st.session_state.vote['item']} because {st.session_state.vote['reason']}"


# import streamlit as st
# import time

# # Sidebar placeholder
# with st.sidebar:
#     spinner_placeholder = st.empty()  # Create a placeholder in the sidebar

# # Simulate a loading action
# if st.button("Start Loading"):
#     with spinner_placeholder.container():
#         st.write("⏳ Loading...")  # Placeholder for spinner-like text
#     time.sleep(3)  # Simulate a delay
#     spinner_placeholder.empty()  # Clear the placeholder after loading

# # Main content
# st.write("Main application content goes here.")




# import asyncio
# import streamlit as st

# # Define an asynchronous task
# async def async_task(delay):
#     await asyncio.sleep(delay)
#     return f"Task completed after {delay} seconds"

# # Streamlit app
# def main():
#     st.title("Simple Asyncio Demo in Streamlit")

#     if st.button("Run Task"):
#         st.write("Task started...")
#         result = asyncio.run(async_task(2))  # Run a simple 2-second task
#         st.write(result)

# if __name__ == "__main__":
#     main()

# import plotly.graph_objects as go
# import pandas as pd

# def create_interactive_table(df: pd.DataFrame, title: str) -> go.Figure:
#     fig = go.Figure(
#         data=[
#             go.Table(
#                 header=dict(
#                     values=list(df.columns),
#                     fill_color='paleturquoise',
#                     align='left'
#                 ),
#                 cells=dict(
#                     values=[df[col] for col in df.columns],
#                     fill_color='lavender',
#                     align='left'
#                 )
#             )
#         ]
#     )
#     fig.update_layout(title=title)
#     return fig


# import chainlit as cl

# @cl.on_chat_start
# async def main():
#     # Sample data
#     data = {
#         'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#         'Age': [25, 30, 35, 40],
#         'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
#     }
#     df = pd.DataFrame(data)

#     # Create the interactive table figure
#     fig = create_interactive_table(df, "Sample Data Table")

#     # Display the table in the chatbot
#     elements = [cl.Plotly(name="Sample Table", figure=fig, display="inline")]
#     await cl.Message(content="Here is the interactive data table:", elements=elements).send()



# import random
# from datetime import datetime
# now = datetime.now()
# num = random.randint(1, 101)

# with open('/tmp/rand.txt', 'a') as f:
#     f.write("{} - Your random number is {}\n".format(now, num))
    


# import streamlit as st
# import time

# # Define the text sections
# large_text = """
# Streamlit is an open-source app framework designed for machine learning and data science projects.
# With Streamlit, developers can create and share custom web apps in just a few minutes. It focuses on simplicity and interactivity, 
# allowing for quick prototyping of ML models and data analysis dashboards.
# """

# remaining_text = """
# Applications of Streamlit:
# 1. Deploy machine learning models for end-users.
# 2. Build dashboards to explore and visualize data.
# 3. Prototype interactive AI/ML experiments effortlessly.
# """

# # Python code to be streamed
# python_code = """
# import streamlit as st

# st.title("Welcome to Streamlit!")
# st.write("This is a sample Python script.")
# """

# # Streamlit App
# st.title("Simulated Streaming Text in Streamlit")

# # Function for streaming text
# def stream_text(text, placeholder, delay=0.008):
#     text_so_far = ""
#     for char in text:
#         text_so_far += char
#         placeholder.text(text_so_far)
#         time.sleep(delay)

# # Function for streaming Python code
# def stream_code(code, placeholder, delay=0.05):
#     code_so_far = ""
#     for char in code:
#         code_so_far += char
#         placeholder.code(code_so_far, language='python')
#         time.sleep(delay)

# # Create a placeholder for the first section
# placeholder1 = st.empty()
# stream_text(large_text, placeholder1)

# # Add an expander and stream the Python code inside it
# with st.status("Generating Python Code..."):
#     code_placeholder = st.empty()
#     stream_code(python_code, code_placeholder)

# st.write("Continuing streamed text:")

# # Stream the remaining text outside the expander
# placeholder2 = st.empty()
# stream_text(remaining_text, placeholder2)



# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import time

# intro_text = """
# Welcome to the Streamlit App showcasing dynamic visualizations. This app uses Python and Plotly for interactive data exploration, including bar charts, scatter plots, and stacked bar charts. Each section provides detailed insights to help you understand and apply the visualizations in real-world scenarios.
# """

# def stream_text(text, placeholder, delay=0.03):
#     text_so_far = ""
#     for char in text:
#         text_so_far += char
#         placeholder.text(text_so_far)
#         time.sleep(delay)

# def stream_code(code, expander, delay=0.005):
#     code_so_far = ""
#     with expander:
#         code_placeholder = st.empty()
#         for token in code:
#             code_so_far += token
#             code_placeholder.code(code_so_far, language="python")
#             time.sleep(delay)

# st.title("Dynamic Visualizations with Plotly")

# placeholder_intro = st.empty()
# stream_text(intro_text, placeholder_intro)

# python_code_bar_chart = """
# categories = ['Category A', 'Category B', 'Category C', 'Category D']
# values = [10, 20, 15, 30]

# fig = go.Figure(data=[
#     go.Bar(x=categories, y=values, marker_color='rgb(118, 185, 0)')
# ])
# fig.update_layout(
#     title='Bar Chart',
#     xaxis_title='Categories',
#     yaxis_title='Values',
#     template='plotly_white'
# )
# st.plotly_chart(fig)
# """
# expander1 = st.expander("View Code: Bar Chart")
# stream_code(python_code_bar_chart, expander1)

# categories = ['Category A', 'Category B', 'Category C', 'Category D']
# values = [10, 20, 15, 30]

# fig = go.Figure(data=[
#     go.Bar(x=categories, y=values, marker_color='rgb(118, 185, 0)')
# ])
# fig.update_layout(
#     title='Bar Chart',
#     xaxis_title='Categories',
#     yaxis_title='Values',
#     template='plotly_white'
# )
# st.plotly_chart(fig)

# insights_bar_chart = """
# Bar charts are effective for comparing discrete categories. This example highlights differences in values across categories, making it ideal for tasks like sales analysis or resource allocation.
# """
# placeholder_explanation1 = st.empty()
# stream_text(insights_bar_chart, placeholder_explanation1)

# python_code_scatter_plot = """
# np.random.seed(42)
# x = np.random.rand(50)
# y = np.random.rand(50)

# fig = go.Figure(data=[
#     go.Scatter(x=x, y=y, mode='markers', marker=dict(color='rgb(118, 185, 0)', size=10))
# ])
# fig.update_layout(
#     title='Scatter Plot',
#     xaxis_title='X-axis',
#     yaxis_title='Y-axis',
#     template='plotly_white'
# )
# st.plotly_chart(fig)
# """
# expander2 = st.expander("View Code: Scatter Plot")
# stream_code(python_code_scatter_plot, expander2)

# np.random.seed(42)
# x = np.random.rand(50)
# y = np.random.rand(50)

# fig = go.Figure(data=[
#     go.Scatter(x=x, y=y, mode='markers', marker=dict(color='rgb(118, 185, 0)', size=10))
# ])
# fig.update_layout(
#     title='Scatter Plot',
#     xaxis_title='X-axis',
#     yaxis_title='Y-axis',
#     template='plotly_white'
# )
# st.plotly_chart(fig)

# insights_scatter_plot = """
# Scatter plots visualize the relationship between two continuous variables. This plot helps identify trends, clusters, or outliers in data.
# """
# placeholder_explanation2 = st.empty()
# stream_text(insights_scatter_plot, placeholder_explanation2)

# python_code_stacked_bar = """
# labels = ['Q1', 'Q2', 'Q3', 'Q4']
# product_a = [15, 25, 35, 45]
# product_b = [10, 20, 25, 30]

# fig = go.Figure(data=[
#     go.Bar(x=labels, y=product_a, name='Product A', marker_color='rgb(118, 185, 0)'),
#     go.Bar(x=labels, y=product_b, name='Product B', marker_color='rgb(39, 174, 96)')
# ])
# fig.update_layout(
#     barmode='stack',
#     title='Stacked Bar Chart',
#     xaxis_title='Quarters',
#     yaxis_title='Sales',
#     template='plotly_white'
# )
# st.plotly_chart(fig)
# """
# expander3 = st.expander("View Code: Stacked Bar Chart")
# stream_code(python_code_stacked_bar, expander3)

# labels = ['Q1', 'Q2', 'Q3', 'Q4']
# product_a = [15, 25, 35, 45]
# product_b = [10, 20, 25, 30]

# fig = go.Figure(data=[
#     go.Bar(x=labels, y=product_a, name='Product A', marker_color='rgb(118, 185, 0)'),
#     go.Bar(x=labels, y=product_b, name='Product B', marker_color='rgb(39, 174, 96)')
# ])
# fig.update_layout(
#     barmode='stack',
#     title='Stacked Bar Chart',
#     xaxis_title='Quarters',
#     yaxis_title='Sales',
#     template='plotly_white'
# )
# st.plotly_chart(fig)

# insights_stacked_bar = """
# Stacked bar charts show how different components contribute to a total value. This example displays sales distribution for two products across four quarters.
# """
# placeholder_explanation3 = st.empty()
# stream_text(insights_stacked_bar, placeholder_explanation3)

# python_code_line_plot = """
# time_points = np.arange(0, 10, 0.1)
# trend = np.exp(time_points / 10)

# fig = go.Figure(data=[
#     go.Scatter(x=time_points, y=trend, mode='lines', line=dict(color='rgb(118, 185, 0)', width=3), name='Trend')
# ])
# fig.add_trace(go.Scatter(
#     x=time_points, y=trend + 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), showlegend=False))
# fig.add_trace(go.Scatter(
#     x=time_points, y=trend - 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), fill='tonexty', showlegend=False))
# fig.update_layout(
#     title='Line Plot with Highlight',
#     xaxis_title='Time',
#     yaxis_title='Value',
#     template='plotly_white'
# )
# st.plotly_chart(fig)
# """
# expander4 = st.expander("View Code: Line Plot with Highlight")
# stream_code(python_code_line_plot, expander4)

# time_points = np.arange(0, 10, 0.1)
# trend = np.exp(time_points / 10)

# fig = go.Figure(data=[
#     go.Scatter(x=time_points, y=trend, mode='lines', line=dict(color='rgb(118, 185, 0)', width=3), name='Trend')
# ])
# fig.add_trace(go.Scatter(
#     x=time_points, y=trend + 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), showlegend=False))
# fig.add_trace(go.Scatter(
#     x=time_points, y=trend - 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), fill='tonexty', showlegend=False))
# fig.update_layout(
#     title='Line Plot with Highlight',
#     xaxis_title='Time',
#     yaxis_title='Value',
#     template='plotly_white'
# )
# st.plotly_chart(fig)

# insights_line_plot = """
# Line plots are ideal for tracking trends over time. The highlighted area shows variability, making it useful for uncertainty analysis in forecasts.
# """
# placeholder_explanation4 = st.empty()
# stream_text(insights_line_plot, placeholder_explanation4)

# final_message = """
# Thank you for exploring dynamic visualizations in Python. Plotly enables interactivity and enhances data storytelling, perfect for creating impactful dashboards.
# """
# placeholder_final = st.empty()
# stream_text(final_message, placeholder_final)




# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# import time

# # Function to display the modal dialog
# @st.dialog("Dynamic Visualizations", width="large")
# def show_visualizations():
#     intro_text = """
#     Welcome to the Streamlit App showcasing dynamic visualizations. This app uses Python and Plotly for interactive data exploration, including bar charts, scatter plots, and stacked bar charts. Each section provides detailed insights to help you understand and apply the visualizations in real-world scenarios.
#     """

#     def stream_text(text, placeholder, delay=0.03):
#         text_so_far = ""
#         for char in text:
#             text_so_far += char
#             placeholder.text(text_so_far)
#             time.sleep(delay)

#     def stream_code(code, expander, delay=0.005):
#         code_so_far = ""
#         with expander:
#             code_placeholder = st.empty()
#             for token in code:
#                 code_so_far += token
#                 code_placeholder.code(code_so_far, language="python")
#                 time.sleep(delay)

#     st.title("Dynamic Visualizations with Plotly")

#     placeholder_intro = st.empty()
#     stream_text(intro_text, placeholder_intro)

#     python_code_bar_chart = """
#     categories = ['Category A', 'Category B', 'Category C', 'Category D']
#     values = [10, 20, 15, 30]

#     fig = go.Figure(data=[
#         go.Bar(x=categories, y=values, marker_color='rgb(118, 185, 0)')
#     ])
#     fig.update_layout(
#         title='Bar Chart',
#         xaxis_title='Categories',
#         yaxis_title='Values',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)
#     """
#     expander1 = st.expander("View Code: Bar Chart")
#     stream_code(python_code_bar_chart, expander1)

#     categories = ['Category A', 'Category B', 'Category C', 'Category D']
#     values = [10, 20, 15, 30]

#     fig = go.Figure(data=[
#         go.Bar(x=categories, y=values, marker_color='rgb(118, 185, 0)')
#     ])
#     fig.update_layout(
#         title='Bar Chart',
#         xaxis_title='Categories',
#         yaxis_title='Values',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)

#     insights_bar_chart = """
#     Bar charts are effective for comparing discrete categories. This example highlights differences in values across categories, making it ideal for tasks like sales analysis or resource allocation.
#     """
#     placeholder_explanation1 = st.empty()
#     stream_text(insights_bar_chart, placeholder_explanation1)

#     python_code_scatter_plot = """
#     np.random.seed(42)
#     x = np.random.rand(50)
#     y = np.random.rand(50)

#     fig = go.Figure(data=[
#         go.Scatter(x=x, y=y, mode='markers', marker=dict(color='rgb(118, 185, 0)', size=10))
#     ])
#     fig.update_layout(
#         title='Scatter Plot',
#         xaxis_title='X-axis',
#         yaxis_title='Y-axis',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)
#     """
#     expander2 = st.expander("View Code: Scatter Plot")
#     stream_code(python_code_scatter_plot, expander2)

#     np.random.seed(42)
#     x = np.random.rand(50)
#     y = np.random.rand(50)

#     fig = go.Figure(data=[
#         go.Scatter(x=x, y=y, mode='markers', marker=dict(color='rgb(118, 185, 0)', size=10))
#     ])
#     fig.update_layout(
#         title='Scatter Plot',
#         xaxis_title='X-axis',
#         yaxis_title='Y-axis',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)

#     insights_scatter_plot = """
#     Scatter plots visualize the relationship between two continuous variables. This plot helps identify trends, clusters, or outliers in data.
#     """
#     placeholder_explanation2 = st.empty()
#     stream_text(insights_scatter_plot, placeholder_explanation2)

#     python_code_stacked_bar = """
#     labels = ['Q1', 'Q2', 'Q3', 'Q4']
#     product_a = [15, 25, 35, 45]
#     product_b = [10, 20, 25, 30]

#     fig = go.Figure(data=[
#         go.Bar(x=labels, y=product_a, name='Product A', marker_color='rgb(118, 185, 0)'),
#         go.Bar(x=labels, y=product_b, name='Product B', marker_color='rgb(39, 174, 96)')
#     ])
#     fig.update_layout(
#         barmode='stack',
#         title='Stacked Bar Chart',
#         xaxis_title='Quarters',
#         yaxis_title='Sales',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)
#     """
#     expander3 = st.expander("View Code: Stacked Bar Chart")
#     stream_code(python_code_stacked_bar, expander3)

#     labels = ['Q1', 'Q2', 'Q3', 'Q4']
#     product_a = [15, 25, 35, 45]
#     product_b = [10, 20, 25, 30]

#     fig = go.Figure(data=[
#         go.Bar(x=labels, y=product_a, name='Product A', marker_color='rgb(118, 185, 0)'),
#         go.Bar(x=labels, y=product_b, name='Product B', marker_color='rgb(39, 174, 96)')
#     ])
#     fig.update_layout(
#         barmode='stack',
#         title='Stacked Bar Chart',
#         xaxis_title='Quarters',
#         yaxis_title='Sales',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)

#     insights_stacked_bar = """
#     Stacked bar charts show how different components contribute to a total value. This example displays sales distribution for two products across four quarters.
#     """
#     placeholder_explanation3 = st.empty()
#     stream_text(insights_stacked_bar, placeholder_explanation3)

#     python_code_line_plot = """
#     time_points = np.arange(0, 10, 0.1)
#     trend = np.exp(time_points / 10)

#     fig = go.Figure(data=[
#         go.Scatter(x=time_points, y=trend, mode='lines', line=dict(color='rgb(118, 185, 0)', width=3), name='Trend')
#     ])
#     fig.add_trace(go.Scatter(
#         x=time_points, y=trend + 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), showlegend=False))
#     fig.add_trace(go.Scatter(
#         x=time_points, y=trend - 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), fill='tonexty', showlegend=False))
#     fig.update_layout(
#         title='Line Plot with Highlight',
#         xaxis_title='Time',
#         yaxis_title='Value',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)
#     """
#     expander4 = st.expander("View Code: Line Plot with Highlight")
#     stream_code(python_code_line_plot, expander4)

#     time_points = np.arange(0, 10, 0.1)
#     trend = np.exp(time_points / 10)

#     fig = go.Figure(data=[
#         go.Scatter(x=time_points, y=trend, mode='lines', line=dict(color='rgb(118, 185, 0)', width=3), name='Trend')
#     ])
#     fig.add_trace(go.Scatter(
#         x=time_points, y=trend + 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), showlegend=False))
#     fig.add_trace(go.Scatter(
#         x=time_points, y=trend - 1, mode='lines', line=dict(color='rgba(118, 185, 0, 0.3)'), fill='tonexty', showlegend=False))
#     fig.update_layout(
#         title='Line Plot with Highlight',
#         xaxis_title='Time',
#         yaxis_title='Value',
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)

#     insights_line_plot = """
#     Line plots are ideal for tracking trends over time. The highlighted area shows variability, making it useful for uncertainty analysis in forecasts.
#     """
#     placeholder_explanation4 = st.empty()
#     stream_text(insights_line_plot, placeholder_explanation4)

#     final_message = """
#     Thank you for exploring dynamic visualizations in Python. Plotly enables interactivity and enhances data storytelling, perfect for creating impactful dashboards.
#     """
#     placeholder_final = st.empty()
#     stream_text(final_message, placeholder_final)

# # Main application
# st.title("Interactive Streamlit Application")

# st.write("Click the button below to view dynamic visualizations in a modal dialog.")

# if prompt:= st.chat_input("Write your lines here..."):
#     st.write("User entered the text")
    
# if st.button("Open Visualizations"):
#     show_visualizations()




# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd

# def Dataset():

#     # Generate the dataset
#     def create_dataset():
#         np.random.seed(42)
#         categories = ['Category A', 'Category B', 'Category C', 'Category D']
#         regions = ['North', 'South', 'East', 'West']
#         data = {
#             "Category": np.random.choice(categories, 100),
#             "Region": np.random.choice(regions, 100),
#             "Sales": np.random.randint(50, 500, 100),
#             "Profit": np.random.randint(10, 100, 100),
#             "Discount": np.random.randint(0, 30, 100),
#             "Quantity": np.random.randint(1, 20, 100),
#         }
#         return pd.DataFrame(data)

#     # Filter the dataset
#     def filter_dataset(df):
#         st.sidebar.header("Filters")
        
#         # Category filter
#         categories = st.sidebar.multiselect("Select Categories", options=df["Category"].unique(), default=df["Category"].unique())
        
#         # Region filter
#         regions = st.sidebar.multiselect("Select Regions", options=df["Region"].unique(), default=df["Region"].unique())
        
#         # Sales range filter
#         min_sales, max_sales = st.sidebar.slider("Sales Range", min_value=int(df["Sales"].min()), max_value=int(df["Sales"].max()), value=(int(df["Sales"].min()), int(df["Sales"].max())))
        
#         # Profit range filter
#         min_profit, max_profit = st.sidebar.slider("Profit Range", min_value=int(df["Profit"].min()), max_value=int(df["Profit"].max()), value=(int(df["Profit"].min()), int(df["Profit"].max())))
        
#         # Apply filters
#         filtered_data = df[
#             (df["Category"].isin(categories)) &
#             (df["Region"].isin(regions)) &
#             (df["Sales"] >= min_sales) &
#             (df["Sales"] <= max_sales) &
#             (df["Profit"] >= min_profit) &
#             (df["Profit"] <= max_profit)
#         ]
#         return filtered_data

#     # Main function logic
#     st.title("Interactive Dataset with Filters")
    
#     # Create the dataset
#     df = create_dataset()
    
#     # Apply filters
#     filtered_df = filter_dataset(df)
    
#     # Display filtered dataset
#     st.subheader("Filtered Data")
#     st.dataframe(filtered_df)
    
#     # Display summary metrics
#     st.subheader("Summary Metrics")
#     st.write(f"Total Sales: ${filtered_df['Sales'].sum():,.2f}")
#     st.write(f"Total Profit: ${filtered_df['Profit'].sum():,.2f}")
#     st.write(f"Total Quantity: {filtered_df['Quantity'].sum()}")

#     # Visualizations
#     st.subheader("Sales by Category")
#     sales_by_category = filtered_df.groupby("Category")["Sales"].sum().reset_index()
#     st.bar_chart(sales_by_category.set_index("Category"))

#     st.subheader("Sales by Region")
#     sales_by_region = filtered_df.groupby("Region")["Sales"].sum().reset_index()
#     st.bar_chart(sales_by_region.set_index("Region"))
    

# # Create a sample dataset
# def get_sample_data():
#     np.random.seed(42)
#     data = {
#         'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 100),
#         'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
#         'Allocation': np.random.randint(50, 200, 100),
#         'Demand': np.random.randint(60, 220, 100),
#         'Supply': np.random.randint(55, 210, 100),
#         'Month': np.random.choice(
#             ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], 
#             100
#         )
#     }
#     return pd.DataFrame(data)

# # Function to create the dashboard
# def Dashboard():
#     st.title("Comprehensive Demand Planner Dashboard")

#     # Load dataset
#     df = get_sample_data()

#     # Sidebar Filters
#     st.sidebar.header("Filters")
#     selected_products = st.sidebar.multiselect("Select Products", df['Product'].unique(), default=df['Product'].unique())
#     selected_regions = st.sidebar.multiselect("Select Regions", df['Region'].unique(), default=df['Region'].unique())
#     selected_months = st.sidebar.multiselect("Select Months", df['Month'].unique(), default=df['Month'].unique())

#     # Apply filters to the dataset
#     filtered_df = df[
#         (df['Product'].isin(selected_products)) &
#         (df['Region'].isin(selected_regions)) &
#         (df['Month'].isin(selected_months))
#     ]

#     # Display filtered dataset
#     st.subheader("Filtered Data")
#     st.dataframe(filtered_df)

#     # Key Metrics
#     st.subheader("Key Metrics")
#     total_allocation = filtered_df['Allocation'].sum()
#     total_demand = filtered_df['Demand'].sum()
#     total_supply = filtered_df['Supply'].sum()
#     demand_fulfillment_rate = (total_supply / total_demand) * 100 if total_demand > 0 else 0
#     supply_accuracy = (total_supply / total_allocation) * 100 if total_allocation > 0 else 0

#     col1, col2, col3 = st.columns(3)
#     col1.metric(label="Total Allocation", value=f"{total_allocation} Units")
#     col2.metric(label="Total Demand", value=f"{total_demand} Units")
#     col3.metric(label="Demand Fulfillment Rate", value=f"{demand_fulfillment_rate:.2f} %")

#     # 1. Grouped Bar Chart: Allocation vs Demand vs Supply
#     st.subheader("1. Product-Level Comparison: Allocation vs Demand vs Supply")
#     grouped_data = filtered_df.groupby('Product').sum().reset_index()
#     fig = go.Figure(data=[
#         go.Bar(name='Allocation', x=grouped_data['Product'], y=grouped_data['Allocation'], marker_color='rgb(118, 185, 0)'),
#         go.Bar(name='Demand', x=grouped_data['Product'], y=grouped_data['Demand'], marker_color='rgb(39, 174, 96)'),
#         go.Bar(name='Supply', x=grouped_data['Product'], y=grouped_data['Supply'], marker_color='rgb(0, 123, 167)')
#     ])
#     fig.update_layout(barmode='group', title="Allocation vs Demand vs Supply", xaxis_title="Products", yaxis_title="Units")
#     st.plotly_chart(fig, use_container_width=True)

#     # 2. Regional Heatmap
#     st.subheader("2. Regional Performance Heatmap")
#     heatmap_data = filtered_df.groupby('Region').sum()[['Allocation', 'Demand', 'Supply']].reset_index()
#     fig = px.imshow(
#         heatmap_data[['Allocation', 'Demand', 'Supply']].values,
#         labels=dict(x="Metrics", y="Region", color="Units"),
#         x=['Allocation', 'Demand', 'Supply'],
#         y=heatmap_data['Region'],
#         color_continuous_scale='Greens'
#     )
#     fig.update_layout(title="Regional Performance Heatmap")
#     st.plotly_chart(fig, use_container_width=True)

#     # 3. Monthly Trends
#     st.subheader("3. Monthly Trends for Allocation, Demand, and Supply")
#     monthly_data = filtered_df.groupby('Month').sum().reset_index()
#     fig = go.Figure(data=[
#         go.Scatter(x=monthly_data['Month'], y=monthly_data['Allocation'], mode='lines+markers', name='Allocation'),
#         go.Scatter(x=monthly_data['Month'], y=monthly_data['Demand'], mode='lines+markers', name='Demand'),
#         go.Scatter(x=monthly_data['Month'], y=monthly_data['Supply'], mode='lines+markers', name='Supply')
#     ])
#     fig.update_layout(title="Monthly Trends", xaxis_title="Month", yaxis_title="Units")
#     st.plotly_chart(fig, use_container_width=True)

#     # 4. Demand Fulfillment by Region
#     st.subheader("4. Demand Fulfillment by Region")
#     region_fulfillment = filtered_df.groupby('Region').sum()
#     region_fulfillment['Fulfillment Rate'] = (region_fulfillment['Supply'] / region_fulfillment['Demand']) * 100
#     fig = px.bar(region_fulfillment.reset_index(), x='Region', y='Fulfillment Rate', title="Demand Fulfillment Rate by Region", template = 'seaborn', color_continuous_scale=px.colors.diverging.Temps)
#     st.plotly_chart(fig, use_container_width=True)

#     # 5. Pareto Chart for Demand Analysis
#     st.subheader("5. Pareto Analysis of Demand")
#     demand_data = filtered_df.groupby('Product').sum().sort_values(by='Demand', ascending=False).reset_index()
#     demand_data['Cumulative Demand'] = demand_data['Demand'].cumsum()
#     demand_data['Cumulative Percentage'] = (demand_data['Cumulative Demand'] / demand_data['Demand'].sum()) * 100
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=demand_data['Product'], y=demand_data['Demand'], name="Demand"))
#     fig.add_trace(go.Scatter(x=demand_data['Product'], y=demand_data['Cumulative Percentage'], name="Cumulative %"))
#     fig.update_layout(title="Pareto Analysis of Demand", xaxis_title="Products", yaxis_title="Demand")
#     st.plotly_chart(fig, use_container_width=True)

#     # 6. Efficiency Gauge
#     st.subheader("6. Efficiency Gauge")
#     efficiency = (total_supply / total_demand) * 100 if total_demand > 0 else 0
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number", value=efficiency, title={'text': "Efficiency (%)"},
#         gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "rgb(118, 185, 0)"}}
#     ))
#     st.plotly_chart(fig, use_container_width=True)

#     # 7. Product Breakdown (Pie Chart)
#     st.subheader("7. Product Breakdown")
#     product_breakdown = filtered_df.groupby('Product').sum().reset_index()
#     fig = px.pie(product_breakdown, values='Demand', names='Product', title="Product Breakdown by Demand")
#     st.plotly_chart(fig, use_container_width=True)

#     # 8. Inventory Turnover (Line Chart)
#     st.subheader("8. Inventory Turnover Over Time")
#     monthly_turnover = (filtered_df.groupby('Month')['Supply'].sum() / filtered_df.groupby('Month')['Demand'].sum()).fillna(0)
#     fig = px.line(x=monthly_turnover.index, y=monthly_turnover.values, labels={"x": "Month", "y": "Turnover"}, title="Inventory Turnover")
#     st.plotly_chart(fig, use_container_width=True)

#     # 9. Allocation-Demand Supply Gap (Heatmap)
#     st.subheader("9. Allocation-Demand-Supply Gap Heatmap")
#     gap = filtered_df.groupby(['Product', 'Region']).sum().reset_index()
#     gap['Gap'] = gap['Allocation'] - gap['Demand']
#     fig = px.density_heatmap(gap, x='Region', y='Product', z='Gap', color_continuous_scale='Viridis', title="Gap Heatmap")
#     st.plotly_chart(fig, use_container_width=True)

#     # 10. Scatter Plot: Allocation vs Demand Correlation
#     st.subheader("10. Correlation Between Allocation and Demand")
#     fig = px.scatter(filtered_df, x='Allocation', y='Demand', color='Product', title="Allocation vs Demand Correlation")
#     st.plotly_chart(fig, use_container_width=True)

# def Chatbot():

#     st.title("Echo Bot")

#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # React to user input
#     if prompt := st.chat_input("What is up?"):
#         # Display user message in chat message container
#         st.chat_message("user").markdown(prompt)
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         response = f"Echo: {prompt}"
#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#             st.markdown(response)
#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})

# pg = st.navigation([st.Page(Dashboard, title="📊 Dashboard"), st.Page(Chatbot, title="💬 Talk to your data")])
# pg.run()


import streamlit as st
import time

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# List of button options
button_list = ["Show Welcome Message", "Display Random Number", "Greet the User", "Show Dataset", "Thank You"]

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



# Display buttons dynamically and check which one is clicked
for button_name in button_list:
    if st.sidebar.button(button_name):
        prompt = button_name
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})




























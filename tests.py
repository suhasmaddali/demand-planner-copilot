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



import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the introductory text
intro_text = """
Welcome to the Streamlit App with dynamic code streaming inside expanders! This demo showcases Python use cases for plotting, 
data analysis, and visualization, with code, execution outputs, and explanations all dynamically presented.
"""

# Code Block 1: Plotting a sine and cosine wave using Plotly
python_code_plot = """
import numpy as np
import plotly.graph_objects as go

# Generate random data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a plotly figure
fig = go.Figure()

# Add sine wave
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Sine Wave', line=dict(color='blue')))

# Add cosine wave
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Cosine Wave', line=dict(color='red')))

# Customize layout
fig.update_layout(
    title='Interactive Sine and Cosine Waves',
    xaxis_title='X-axis',
    yaxis_title='Y-axis',
    template='plotly_dark',
    showlegend=True
)

# Show the plot in Streamlit
st.plotly_chart(fig)
"""

# Code Block 2: Simple DataFrame creation and display
python_code_dataframe = """
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Score': [85, 90, 95, 80]
}

df = pd.DataFrame(data)

# Display the DataFrame in Streamlit
st.write('Here is a sample DataFrame:')
st.dataframe(df)
"""

# Code Block 3: Matplotlib visualization
python_code_matplotlib = """
import numpy as np
import matplotlib.pyplot as plt

# Generate random data for histogram
data = np.random.randn(1000)

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot in Streamlit
st.pyplot(plt)
"""

# Function for streaming text
def stream_text(text, placeholder, delay=0.05):
    text_so_far = ""
    for char in text:
        text_so_far += char
        placeholder.text(text_so_far)
        time.sleep(delay)

# Function for streaming Python code (one token at a time)
def stream_code(code, expander, delay=0.005):
    code_so_far = ""
    with expander:
        code_placeholder = st.empty()
        for token in code:
            code_so_far += token
            code_placeholder.code(code_so_far, language="python")
            time.sleep(delay)

# Streamlit App
st.title("Dynamic Code Streaming in Expanders")

# Stream the introduction
placeholder_intro = st.empty()
stream_text(intro_text, placeholder_intro)

expander1 = st.status("View Code for Plotting with Plotly")
stream_code(python_code_plot, expander1)

# Execute the first code block
exec(python_code_plot)

# Stream explanation after execution
explanation1 = """
This plot demonstrates the sine and cosine waves generated using NumPy and Plotly. 
The interactive nature of the plot lets you explore the data effectively.
"""
placeholder_explanation1 = st.empty()
stream_text(explanation1, placeholder_explanation1)

expander2 = st.status("View Code for Data Analysis with Pandas")
stream_code(python_code_dataframe, expander2)

# Execute the second code block
exec(python_code_dataframe)

# Stream explanation after execution
explanation2 = """
This example shows how to create and display a DataFrame using Pandas. 
It's a powerful tool for working with structured data in Python.
"""
placeholder_explanation2 = st.empty()
stream_text(explanation2, placeholder_explanation2)

expander3 = st.status("View Code for Visualization with Matplotlib")
stream_code(python_code_matplotlib, expander3)

# Execute the third code block
exec(python_code_matplotlib)

# Stream explanation after execution
explanation3 = """
This histogram visualizes the distribution of random data using Matplotlib. 
Such visualizations help in understanding data distributions and identifying patterns.
"""
placeholder_explanation3 = st.empty()
stream_text(explanation3, placeholder_explanation3)

# Final message
final_message = """
Thank you for exploring this interactive demo! Streamlit makes it seamless to combine Python code, interactive visualizations, 
and dynamically streamed explanations into a single app.
"""
placeholder_final = st.empty()
stream_text(final_message, placeholder_final)












import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Set up the Groq API client.
load_dotenv()

st.markdown(
    """
    <style>
    .title {
        font-size: 3em;
        font-weight: bold;
        color: #3a6f69;
        text-align: center;
        margin-bottom: 20px;
        font-family: 'Georgia'; 
    }
    .subtitle {
        color: #47938a;
        text-align: center;
        font-weight: bold;
        font-family: 'Georgia'; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Pandas AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions about diseases and their symptoms</div>',
    unsafe_allow_html=True,
)
# Create a text input widget for the user to enter their question.
question = st.text_input("Enter your question:")

# If the user has entered a question, create a Pandas AI Agent and use it to answer the question.
if question:
    df = pd.read_csv("diseases.csv")

    # Create a Pandas AI Agent.
    agent = create_pandas_dataframe_agent(ChatGroq(model_name="llama3-70b-8192"), df, verbose=True, allow_dangerous_code=True, handle_tool_error=True) #added handle_tool_error

    # Use the agent to answer the question.
    answer = agent.run(question)

    # Print the answer to the Streamlit app.
    st.write(answer)
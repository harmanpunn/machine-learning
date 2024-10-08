import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm, api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")
api_key = st.sidebar.text_input("Enter your Open AI API Key", type="password")

## Drop down to select various OpenAI models
llm = st.sidebar.selectbox("Select the OpenAI model", ["gpt-4-turbo", "gpt-4"])

# Adjust the response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=200, value=100)

## Main interface for user input
st.write("Please enter your question below:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please enter your question above")
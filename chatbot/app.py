from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")


#prompt template
prompt= ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Please respond to the user queeries."),
     ("user", "Question: {questions}")]
)


#streamlit framework

st.title("Langchain Demo With Huggingface")
input_text= st.text_input("Search the topic you want:")


#defining model
llm_1= HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    temperature= 0.6
)
llm= ChatHuggingFace(llm= llm_1)
output_parser= StrOutputParser()

#chaining
chain= prompt| llm| output_parser

if input_text:
    st.write(chain.invoke({"questions": input_text}))
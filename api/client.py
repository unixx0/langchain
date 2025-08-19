import requests
import streamlit as st

def get_huggingface_respose(input_text):
    response= requests.post("http://localhost:8000/essay/invoke",
    json= {"input": {"topic": input_text}})
    return response.json()["output"]["content"]

def ollama_response(input_text):
    response= requests.post("http://localhost:8000/poem/invoke",
                            json= {"input": {"topic": input_text}})
    return response.json()["output"]


st.title("langchain Demo with llama 2 api: ")
input_text= st.text_input("write an essay: ")
input_text1= st.text_input("Write a poem on:")

if input_text:
    st.write(get_huggingface_respose(input_text))

if input_text1:
    st.write(ollama_response(input_text1))
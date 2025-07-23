import streamlit as st
from function import input
st.header("HR Policy QnA")
input_text= st.text_input("Write any queies related to the office")
if input_text:
    result= input(input_text)
    st.write(result)

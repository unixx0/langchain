from langchain_chroma import Chroma
from langchain_text_splitters import HTMLSemanticPreservingSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
loader= WebBaseLoader(web_path="https://tsukinp.com/" )
document= loader.load()

splitter= RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size= 25, chunk_overlap= 5
)
docs= splitter.split_documents(document)

verctorstore= Chroma(
    embedding_function= HuggingFaceEmbeddings()
)

verctorstore.add_documents(docs)
retriever= verctorstore.as_retriever(search_kwargs={"k": 1})

st.header("TSUKI INFO")
input_text= st.text_input("What is the thing you wanna know about tsuki?")
query= input_text
result= retriever.invoke(query)
st.write(result[0].page_content)

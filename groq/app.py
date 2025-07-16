import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]= os.getenv("groq_api_key")

if "vectors" not in st.session_state:
    st.session_state.embeddings= OllamaEmbeddings(model= 'llama2')
    st.session_state.loader= WebBaseLoader(["https://docs.smith.langchain.com/"])
    st.session_state.docs= st.session_state.loader.load()

    st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
    st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors= FAISS.from_documents(st.session_state.final_documents[:50],st.session_state.embeddings )


st.title("ChatGroq DEMO")
llm= ChatGroq(
    model_name= "gemma2-9b-it"
)


prompt= ChatPromptTemplate.from_template(
    """
You are an intelligent chatbot.
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question: {input}
"""
)
document_chain= create_stuff_documents_chain(llm= llm,
                                             prompt= prompt)
retriever= st.session_state.vectors.as_retriever()
retrieval_chain= create_retrieval_chain(retriever, document_chain)

prompt= st.text_input("Import your prompt here")

if prompt:
    start= time.process_time()
    response= retrieval_chain.invoke({"input": prompt})
    print(f"Response Time: {time.process_time()-start}")
    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("Document Similiarity Search"):
        #find relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

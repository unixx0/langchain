from langchain.retrievers import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

loader= PyPDFLoader(
    file_path= "/home/yunish/Desktop/Langchain/retrievers/healthy_living.pdf"

)

document= loader.load()

splitter= SemanticChunker(embeddings= HuggingFaceEmbeddings())
splitted_docs= splitter.split_documents(document)

vector_store= Chroma.from_documents(documents= splitted_docs, embedding= HuggingFaceEmbeddings())

model= HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    temperature= 0.6
)
llm= ChatHuggingFace(llm=  model)

normal_retriever=  vector_store.as_retriever(
                                         search_kwargs= {"k": 3})

retriever= MultiQueryRetriever.from_llm(
    retriever=normal_retriever,
    llm= llm
)

query= "How can I improve my eating habit?"
result= retriever.invoke(query)
print(result)
print(len(result))

result= normal_retriever.invoke(query)
print(result, len(result))

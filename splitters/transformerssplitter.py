import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_kEY"]= os.getenv("google_api_key")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.chains.query_constructor import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever, AttributeInfo
from langchain_experimental.text_splitter import SemanticChunker

loader= JSONLoader(
file_path= "/home/yunish/Desktop/yunish/Langchain/HR_Policy/data.json",
jq_schema= ".",
text_content= False)#load all data of json file, inshort is a filter
docs= loader.load()

splitter= SentenceTransformersTokenTextSplitter(chunk_overlap= 0)

no_of_tokens= splitter.count_tokens(docs[0].page_content)
print(f"No. of tokens: {no_of_tokens}")

splitted_json= splitter.split_documents(docs)
print(splitted_json)

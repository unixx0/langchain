import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_kEY"]= os.getenv("google_api_key")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.chains.query_constructor import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever, AttributeInfo
from langchain_experimental.text_splitter import SemanticChunker
#load JSON files

loader= JSONLoader(
file_path= "/home/yunish/Desktop/yunish/Langchain/HR_Policy/data.json",
jq_schema= ".",
text_content= False)#load all data of json file, inshort is a filter
docs= loader.load()
        


#break into chunks
text_splitter= SemanticChunker(HuggingFaceEmbeddings())
chunked_docs= text_splitter.split_documents(docs)
print(chunked_docs)
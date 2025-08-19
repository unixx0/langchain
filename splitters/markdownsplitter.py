import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_kEY"]= os.getenv("google_api_key")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.chains.query_constructor import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever, AttributeInfo
from langchain_experimental.text_splitter import SemanticChunker



markdown_text= "# Foo \n \n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

headers_tosplit_on= [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),]

splitter= MarkdownHeaderTextSplitter(headers_tosplit_on, strip_headers= True, return_each_line= True)
splitted= splitter.split_text(markdown_text)
print(splitted)




#Stringing compressors and document transformers together
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, DocumentCompressorPipeline #can be used for using multiple compressors
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]= os.getenv("HUGGINGFACEHUB_API_TOKEN")

#function to print result
def display(result):
    for i in range (len(result)):
        print(f"Result [{i+1}]")
        print(result[i].page_content)

loader= TextLoader(
    file_path= "retrievers/chatgpt.txt"

)
document= loader.load()

splitter= RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 50)
splitted_docs= splitter.split_documents(document)

vectordb= Chroma(
    embedding_function= HuggingFaceEmbeddings()

)

vectordb.add_documents(splitted_docs)

retriever=  vectordb.as_retriever(search_type= "mmr",
                                  search_kwargs= {"k": 5, "lambda_mult": 0.5})

model= HuggingFaceEndpoint(
    model= "openai/gpt-oss-20b"
)

llm= ChatHuggingFace(
    llm= model
)

extractor= LLMChainExtractor.from_llm(llm)
filter= LLMChainFilter.from_llm(llm)

pipeline_compressor= DocumentCompressorPipeline(
    transformers= [extractor, filter]     #llmchainextractor works first followd by llmchainfilter, we can also add basedocument transmitter like textsplitter here
)

contextual_retriever= ContextualCompressionRetriever(
    base_compressor= pipeline_compressor,
    base_retriever= retriever
)

result= contextual_retriever.invoke("What are the purposes of chatgpt?")
display(result)
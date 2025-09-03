from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loaders= [PyPDFLoader(file_path= "retrievers/2022 Q3 AAPL.pdf"),
           PyPDFLoader(file_path= "retrievers/2022 Q3 AMZN.pdf")]

document=[]

for x in loaders:
    document.extend(x.load())



#Retriving full document
#for retriving full document we only require child splitter, parent docs are stored in docstore

splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)

vectorstore= Chroma(embedding_function= HuggingFaceEmbeddings())

store= InMemoryStore()

retriever= ParentDocumentRetriever(
    vectorstore= vectorstore,
    docstore= store,
    child_splitter= splitter
)

retriever.add_documents(documents= document)

#chunked_result= vectorstore.similarity_search(query="How has Apple's total net sales changed over time?")
#print(chunked_result[0].page_content)


docs_result= retriever.invoke("How has Apple's total net sales changed over time?")
print(docs_result)
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



#Retriving larger
"""Sometimes, the full documents can be too big to want to retrieve them as is. In that case, what we really want to do is to first split the raw documents into larger chunks, and then split it into smaller chunks. We then index the smaller chunks, but on retrieval we retrieve the larger chunks (but still not the full documents. Larger chunks are stored in docstore"""

parent_splitter= RecursiveCharacterTextSplitter(chunk_size= 1000)
child_splitter= RecursiveCharacterTextSplitter(chunk_size= 400, chunk_overlap= 200)

vectorstore= Chroma(embedding_function= HuggingFaceEmbeddings())

store= InMemoryStore()

retriever= ParentDocumentRetriever(
    vectorstore= vectorstore,
    docstore= store,
    child_splitter= child_splitter,
    parent_splitter= parent_splitter
)

retriever.add_documents(documents= document)

docs_result= retriever.invoke("How has Apple's total net sales changed over time?")
print(docs_result[0].page_content)

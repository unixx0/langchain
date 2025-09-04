from langchain.retrievers import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

loaders= [PyPDFLoader(file_path= "retrievers/2022 Q3 AAPL.pdf"),
           PyPDFLoader(file_path= "retrievers/2022 Q3 AMZN.pdf")]

document=[]

for x in loaders:
    document.extend(x.load())

vectordb= Chroma(embedding_function= HuggingFaceEmbeddings())

#Storage layer for parent document
store= InMemoryByteStore()
id_key= "doc_id"

retriever= MultiVectorRetriever(
    vectorstore= vectordb,
    byte_store= store,
    id_key= id_key  #must match with key used in line 46
)


child_text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 1000
)


#for generating id for each parent doc which helps linking child chunk with parent doc
doc_ids= [str(uuid.uuid4()) for _ in document]  


#linking the id of child chunk into parent chunk
sub_docs= []   #for storing chunked doc which is linked with parent doc id

for i, doc in enumerate(document):
    _id= doc_ids[i]
    _sub_docs= child_text_splitter.split_documents(documents= [doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key]= _id   #creates a new key named "doc_id" with value _id in metadata of chunked document
    sub_docs.extend(_sub_docs)


retriever.vectorstore.add_documents(sub_docs)

#store parent document in byte store
"""The zip() function in Python pairs elements from two lists (or any iterables) into tuples.

mset stands for “multiple set”.

It’s a method in InMemoryByteStore (and other LangChain stores) that allows you to store multiple key-value pairs at once.

Each key is usually a unique ID, and the value is a Document (or any object you want to store)."""

retriever.docstore.mset(list(zip(doc_ids, document)))

#retriving small chunks
result= retriever.vectorstore.similarity_search(query= "How has Apple's total net sales changed over time?", k=1)
print(result[0].page_content)

#retriving parent document
result= retriever.invoke("How has Apple's total net sales changed over time?")
print(result[1].page_content, len(result))
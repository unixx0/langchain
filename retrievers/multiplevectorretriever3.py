# for Hypothetical Queries
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import MultiVectorRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import InMemoryByteStore
import uuid
from langchain_core.documents import Document

os.environ["GOOGLE_API_KEY"]= os.getenv("google")

class structure(BaseModel):
    questions: list[str]= Field(..., description= "3 questions related to the topic")


loaders= [PyPDFLoader(file_path= "retrievers/2022 Q3 AAPL.pdf"),
          PyPDFLoader(file_path= "retrievers/2022 Q3 AMZN.pdf")]

documents=[]

for x in loaders:
    documents.extend(x.load())


model= ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash"
)

llm= model.with_structured_output(structure)

prompt= ChatPromptTemplate.from_template(
    """Generate any 3 hypothetical questions related to the following document
    <document>
    {docs}
    </document
    """
)


chain= {"docs": lambda x: x.page_content}| prompt| llm

result= chain.batch(documents, {"max_concurrency": 5})

list_questions= [x.questions for x in result]

vectordb= Chroma(embedding_function= HuggingFaceEmbeddings() )

retriever= MultiVectorRetriever(
    vectorstore= vectordb,
    id_key= "doc_id",
    byte_store= InMemoryByteStore()

)

docs_ids= [str(uuid.uuid4()) for  _ in documents]

docs= []
for i, x in enumerate(list_questions):
    for qn in x:
        docs.append(
            Document(page_content= qn,
                     metadata= {"doc_id": docs_ids[i]})
        )

retriever.vectorstore.add_documents(docs)

retriever.docstore.mset(list(zip(docs_ids, documents)))

result= retriever.invoke("How has Apple's total net sales changed over time?")
print(result[0].page_content)


#Associating summaries with a document for retrieval

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
import uuid
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["HUGGINGFACEHUB_API_KEY"]= os.getenv("hugging")

loaders= [PyPDFLoader(file_path= "retrievers/2022 Q3 AAPL.pdf"),
          PyPDFLoader(file_path= "retrievers/2022 Q3 AMZN.pdf")]

documents=[]

for x in loaders:
    documents.extend(x.load())

model= HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
    
)
llm= ChatHuggingFace(llm=  model)

prompt= ChatPromptTemplate.from_template(
    """Generate the summary of following document:
    <document>
    {docs}
    </document>
    """
)

summary_chain= {"docs": lambda x: x.page_content}| prompt | llm| StrOutputParser()

#.batch helps to invoke multiple input parallelly and takes list as input
summary= summary_chain.batch( inputs= documents, kwargs= {"max_concurrency": 5})

doc_ids= [str(uuid.uuid4()) for _ in documents]

vectorstore= Chroma(embedding_function= HuggingFaceEmbeddings())

retriever= MultiVectorRetriever(
    vectorstore= vectorstore,
    byte_store= InMemoryByteStore(),
    id_key= "doc_id",
    search_type= "mmr"


)

summary_document= []
for i, x in enumerate(summary):
    summary_document.append(
        Document(page_content= x,
                 metadata= {"doc_id": doc_ids[i]})
    )


retriever.vectorstore.add_documents(documents= summary_document)
retriever.docstore.mset(list(zip(doc_ids, documents)))

result= retriever.invoke("How has Apple's total net sales changed over time?")
print(result[0].page_content)

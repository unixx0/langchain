from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]= os.getenv("hugging")

model= HuggingFaceEndpoint(
    model= "mistralai/Mistral-7B-Instruct-v0.2",
    temperature= 0.6
)
llm= ChatHuggingFace(llm=  model)

loader= PyPDFLoader(file_path= "retrievers/2022 Q3 AAPL.pdf")
documents= loader.load()

metadata_field_info = [
    AttributeInfo(
        name="author",
        description="The entity or organization that authored the filing (e.g., 'EDGAR Online')",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="The document title or unique identifier (e.g., SEC accession number)",
        type="string",
    ),
    AttributeInfo(
        name="subject",
        description="A description of the filing, such as the form type (10-Q, 10-K) and reporting period",
        type="string",
    ),
    AttributeInfo(
        name="keywords",
        description="Keywords associated with the filing, often including form type or accession number",
        type="string",
    ),
    AttributeInfo(
        name="creationdate",
        description="The date and time when the filing was created",
        type="date",
    ),
    AttributeInfo(
        name="moddate",
        description="The date and time when the filing was last modified",
        type="date",
    ),
    AttributeInfo(
        name="total_pages",
        description="The total number of pages in the filing",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="The file path or source location of the filing (e.g., PDF file name)",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page number within the filing",
        type="integer",
    ),
    AttributeInfo(
        name="page_label",
        description="The label printed on the page (may differ from numerical index)",
        type="string",
    ),
]
document_content_description= "Document regarding Apple.inc"
splitter= RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 100)
docs= splitter.split_documents(documents= documents)

db= Chroma(embedding_function= HuggingFaceEmbeddings())
db.add_documents(documents= docs)

retriever= SelfQueryRetriever.from_llm(
    llm= llm,
    vectorstore= db,
    metadata_field_info= metadata_field_info,
    document_contents= document_content_description,
    enable_limit= True
)

result= retriever.invoke("Give me the  summary of Basis of Presentation and represenatation",  k=1)
print(result[0].page_content)



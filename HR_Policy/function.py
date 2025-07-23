import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_kEY"]= os.getenv("google_api_key")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.chains.query_constructor import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever, AttributeInfo

#load JSON files
def input(input: str) -> str:
    try:
        loader= JSONLoader(
        file_path= "C:\\Users\\sthay\\OneDrive\\Desktop\\Langchain\\HR_Policy\\data.json",
        jq_schema= ".",
        text_content= False)#load all data of json file, inshort is a filter
        docs= loader.load()
        

    except FileNotFoundError as e:
        print("error", e)  


    #break into chunks
    text_splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
    chunked_docs= text_splitter.split_documents(docs)


    #embeed it and store in chroma(vectorstore)
    db= Chroma.from_documents(chunked_docs, HuggingFaceEmbeddings())

    #create metadata_field_info for Self query retriver
    metadata_field_info= [
        AttributeInfo(
            name= "id",
            description= "The unique identifier of policy",
            type= "string"

        ),
        AttributeInfo(
            name= "title",
            description= "Titile of policy",
            type="string"
        ),

        AttributeInfo(
        name="effective_from",
        description="Date when the policy became effective, in YYYY-MM-DD format",
        type="string" 
        ),

        AttributeInfo(
        name="version",
        description="Version number of the policy",
        type="string"
        ),

        AttributeInfo(
        name="status",
        description="Current status of the policy, such as active or inactive",
        type="string"
        ),

        AttributeInfo(
            name= "company",
            description= "The name of company or organization",
            type= "string"
        ),

        AttributeInfo(
            name= "last_updated",
            description= "The recent date when this policy was updated or modified",
            type= "string"

        ),
        
        AttributeInfo(
            name= "contact_person",
            description= "Name, email and role of the person I should contact if I have any queries",
            type= "string"
        )

    ]


    #initialize model
    llm= ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash"
    )

    #create self query retriever
    retriever= SelfQueryRetriever.from_llm(
        llm= llm,
        vectorstore= db,
        metadata_field_info= metadata_field_info,
        document_contents= "Company Policies and Guidelines"
        

    )


    #create prommpt
    from langchain_core.prompts import ChatPromptTemplate
    prompt= ChatPromptTemplate.from_template(
        """
    Answer the given question based on the provided context only.
    Think step by step before providing a detailed answer. If you donot know answer reply "I dont know"
    <context>
    {context}
    </context>
    Question: {input}
    """
    )

    #create document chain
    document_chain= create_stuff_documents_chain(llm, prompt)

    #create retrieval chain that combine retrieval and document chain
    retrieval_chain= create_retrieval_chain(retriever, document_chain)

    response= retrieval_chain.invoke({"input": input})
    return (response["answer"])


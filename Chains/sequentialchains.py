from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")


model= HuggingFaceEndpoint(
    model= "openai/gpt-oss-120b",
    
)
llm= ChatHuggingFace(
    llm= model
)

parser= StrOutputParser()

prompt1= ChatPromptTemplate.from_messages([
    ("system", "You are a intelligent Teacher. Describe about the topic given by the user in a paragraph"),
    ("human", "Topic: {topic}")
]

)
prompt2= ChatPromptTemplate.from_messages([
    ("system", "You are a intelligent Teacher. Now summarize the given paragraph in 5 bullets "),
    ("human", "paragraph: {paragraph}")
]
)

chain  = prompt1| llm| parser| prompt2| llm| parser
output= chain.invoke({"topic": "Blackhole"})
print(output)



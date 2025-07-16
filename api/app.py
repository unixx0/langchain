from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

app= FastAPI (
    title= "Langchain Server",
    version= "1.0",
    description= "A simple API server"
)



llm= HuggingFaceEndpoint(
    repo_id= "mistralai/Mistral-7B-Instruct-v0.2"
)
model1= ChatHuggingFace(llm= llm)
model2= OllamaLLM(model= "llama2")

prompt1= ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)
prompt2= ChatPromptTemplate.from_template(

    "write me an poem about {topic} with 50 words"
)


add_routes(
    app,
    prompt1| model1,
    path= "/essay"
)

add_routes(

    app,
    prompt2| model2,
    path= "/poem"
)

if __name__=="__main__":
    uvicorn.run (app, host= "localhost", port= 8000)
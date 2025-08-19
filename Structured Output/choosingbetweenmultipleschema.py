import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]= os.getenv("google_api_key")
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Union
from langchain_core.messages import HumanMessage
class structure1(BaseModel):
    """use this whenever user is talking about any product"""
    company_name: str= Field(description="Enter the name of the company")
    verdict: str= Field(description= "Your final verdict. Whether the product is good or not")

class structure2(BaseModel):
    "use this whenever user is asking for regarding any GK questions"
    Question: str= Field(description="The question of the user")
    Answer: str= Field(description="Your annswer with respect to the question asked by the user. Give the exact answer without description")

#creatiing a class that combines both the structures
class structure(BaseModel):
    final_output: Union[structure1, structure2]

model= ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash"
)
structured_model= model.with_structured_output(structure, include_raw= True, method= "pydantic")    #include rw and method is optional

result = structured_model.invoke([HumanMessage(content="What is the capital of Nepal")])
print(result["raw"])  #result is divided into raw annd parsed if we use include_raw
print(result["parsed"])
#print(result.final_output)
"""
xyz= structure2(Question= "Hello", Answer= "HII")
print(xyz.model_dump())
"""



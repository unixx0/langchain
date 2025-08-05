from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]= os.getenv("google_api_key")
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from typing import Optional

class person(BaseModel):
    name: str= Field(..., description="The name of person"),   #... means it is compulsary
    age: int=  Field(..., description="The age of person"),
    hobbies: Optional[list[str]]= Field(..., description= "The hobbies of that person")

parser= PydanticOutputParser(pydantic_object= person)
model= ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash"
)
prompt= PromptTemplate(
    template= "Generate a imaginary person and give its details \n {format_instruction}",
    input= [],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
chain= prompt| model | parser
result= chain.invoke({})
print(result)
print(type(result))

print(result.age, type(result.age))
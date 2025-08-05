from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]= os.getenv("google_api_key")
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

model= ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash"
)

#define schema that output parser will follow
schema= [
    ResponseSchema(name= "Fact 1", description="The fact of the given topic"),
    ResponseSchema(name= "Fact 2", description="The fact of the given topic"),
    ResponseSchema(name= "Fact 3", description="The fact of the given topic"),
]
#parser= StructuredOutputParser(response_schemas=schema) OR,
parser= StructuredOutputParser.from_response_schemas(schema)
template= PromptTemplate(
    template= "Tell me 3 different and interesting facts about {topic} \n {format_instruction}",
    input_variable= ["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain= template| model| parser
result= chain.invoke({"topic": "Blackhole"})
print(result)
print(type(result))

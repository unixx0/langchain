from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]= os.getenv("google_api_key")
from langchain_core.output_parsers import JsonOutputParser

model= ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash"
)
parser= JsonOutputParser()
template= PromptTemplate(
    template= "Give me a description about {input} on the basis of name, related_to and description \n {format_instruction} ", #format instruction template ma hunei parcha. yesley instructions lai format garna help garcha while using jsonoutparser and structuredoutputparser
    input= ["input"],
    partial_variables={"format_instruction": parser.get_format_instructions()}

)
chain = template| model| parser
result= chain.invoke({"input": "Blackhole"})
print(result)

"""
OR
IF WE DONT CREATE CHAINS
prompt= template.format(input= "blackhole")
result= model.invoke(prompt)
final_result= parser.parse(result.content)
print(result)
"""
print(type(result))
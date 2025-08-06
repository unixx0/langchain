from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import  RunnableBranch #runnable branch helps to execute chains in if, else
from langchain.schema.runnable import RunnableLambda #takes lambda function as a argument and make it runnable ie. make it chain
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"]= os.getenv("google_api_key")


parser1= StrOutputParser()

class sentiment(BaseModel):   #schema for pydantic output parser
    sentiment: Literal["Positive", "Negative"]= Field(description= "The sentiment of the user response about the product"),



parser2= PydanticOutputParser(pydantic_object= sentiment) #llm generates output in structure format i.e if we use parser2 in chain then llm will generate sentiment which is wither positive or negative in this code


model= GoogleGenerativeAI(
    model= "gemini-2.0-flash"
)


#prompt to generate the sentiment of feedback either Positive or negative using pydanticoutputparser
prompt1= PromptTemplate (
    template= "Give the sentiment of the given feedback by user. feedback -> {feedback} \n and follow this {format_instructions} format",
    input_variables= ["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)
classifier_chain= prompt1| model| parser2
print(classifier_chain.invoke({"feedback": "The app is crashing every time I try to upload a photo! This is completely unusable"}))


#prompt to generate reponse for positive feedback
prompt2= ChatPromptTemplate.from_template(
    "Generate a  response/reply for given positive user feedback \n feedback -> {feedback}"
)

#prompt to generate response for negative feedback
prompt3= ChatPromptTemplate.from_template(
    "Generate a  response/reply for given negative user feedback \n feedback -> {feedback}"
)


"""
syntax= RunnableBranch(
        (condition1, chain1),
        (condition2, chain2),
        (condition N, chain N),
        default_chain

)

Here the runnable branch consist of multiple tuples with condition and respective chains to execute.
If the given condition returns True then it will execute the chain corresponding to the respective condition.
Eg: If Condition1 is met, it executes chain1.
If none of the conditions are True, it will return default chain.

"""

branch_chain= RunnableBranch(
    (lambda x: x.sentiment=="Positive", prompt2|model| parser1 ),
    (lambda x: x.sentiment== "Negative", prompt3|model| parser1),
    RunnableLambda(lambda x: "Invalid sentiment")      #Default chain
)

final_chain= classifier_chain| branch_chain


result= final_chain.invoke({"feedback": "The app is crashing every time I try to upload a photo! This is completely unusable"})

print(result)


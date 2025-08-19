import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]= os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm= HuggingFaceEndpoint(
    model= "mistralai/Devstral-Small-2505",
    temperature= 0.6)

def generate_restaurant_name(cuisine: str):
    #creating template
    restaurant_name= PromptTemplate(
        input_variables= ["cuisine"],
        template= "I want to open a resturant. Please suggest me one {cuisine} restaurant name"
    )
    menu_items= PromptTemplate(
        input_variables= ["items"],
        template= "Please suggest me 5 items that can be in menu  of a {items} restaurant"

    )

    #chaining
    restaurant_name_chain= restaurant_name | llm
    restaurantname= restaurant_name_chain.invoke(cuisine)


    menu_items_chain= menu_items | llm
    menuitems= menu_items_chain.invoke(restaurantname)



    return{
        "restaurant_name": restaurantname,
        "menu_items": menuitems
    }


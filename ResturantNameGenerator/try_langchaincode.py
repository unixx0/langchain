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

def generate_restaurant_name(cuisine):
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
    menu_items_chain= menu_items| llm

    
    #creating sequential chains, transfer output of resturanr_name_chain to menu_items_chain
    def restaurant_name_chain_to_menu_item_chain(restaurant_name: str):
        print(f"\n Restaurant Name: {restaurant_name}")
        return {"items":restaurant_name}

    def menu_item_output(menu_items: str):
        print(f"\n Menu items: {menu_items}")
        return menu_items


    final_chain= restaurant_name_chain| RunnableLambda(restaurant_name_chain_to_menu_item_chain) | menu_items_chain| RunnableLambda(menu_item_output)
    final_chain.invoke({"cuisine": cuisine})
    

if __name__=="__main__":
    generate_restaurant_name("Italian")
        
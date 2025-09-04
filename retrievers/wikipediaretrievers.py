from langchain_community.retrievers import WikipediaRetriever

retriever= WikipediaRetriever(lang= 'en', top_k_results= 5, load_all_available_meta= False)
result= retriever.invoke("What is the GDP of Nepal?")
print(result)
print(type(result))
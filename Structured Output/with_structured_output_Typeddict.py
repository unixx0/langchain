#simple TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm= ChatGroq(
    model_name= "gemma2-9b-it"
)

#create schema for our dict named Review
class Review(TypedDict):

    summary: str
    sentiment: str

structured_llm= llm.with_structured_output(Review)
result= structured_llm.invoke("""I’ve been using the Galaxy S24 for a couple of months now, and I’m really impressed. The display is super smooth and bright, and the phone feels fast no matter what I’m doing. I love the compact size — it’s easy to hold and use with one hand. The camera takes great photos, especially in good lighting. Battery life is decent, though I wish it charged a bit faster. Overall, it’s a solid flagship that feels premium and reliable.""")
print(result)

print("\n")

#Annotated TypedDict: Used to give information to llm about what does the key mean 
from typing import Annotated

class Review (TypedDict):
    key_themes: Annotated[list[str], "Write down all the ky feature pf product"]
    cons: Annotated[list[str], "Write down all the bad things of product"]
    summary: Annotated[str, "A brief summary of review"]
    sentiment: Annotated[str, "Return sentiment of review either negative, positive or neutral" ]

structured_llm= llm.with_structured_output(Review)

result= structured_llm.invoke("""After a month of using the Galaxy S24 Ultra, I’m genuinely impressed with its performance—Snapdragon 8 Gen 3 delivers blazing-fast multitasking, and the 120Hz AMOLED display is stunning in both sunlight and dark modes. The AI-powered camera excels in low light, though edge detection in portrait mode can be inconsistent. Battery life comfortably lasts a day and a half with heavy use, but the phone does get noticeably warm during gaming. OneUI feels refined, though still cluttered compared to stock Android. Overall, it's a powerhouse with minor quirks that serious users might overlook for the sheer capability""")

print(result)

print("\n")



'''using literal and optional
optional: Is optional(may or maynt print). Donot provide any value if there is no any information about that particular key in prompt else print
literal: used to choose value within the list'''

from typing import Literal, Optional

class Review (TypedDict):
    key_themes: Annotated[list[str], "Write down all the ky feature pf product"]
    pros: Annotated[Optional[list[str]], "Write all the good things of this product"]
    cons: Annotated[Optional[list[str]], "Write down all the bad things of product"]
    summary: Annotated[str, "A brief summary of review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of review either negative or positive" ]  #it whill choose from pos or neg
    name: Annotated[Optional[str], "Write the name of the person who gave this review"]  #it will not be shown in the output if the name of reviewer is not given

structured_llm= llm.with_structured_output(Review)

result= structured_llm.invoke("""After a month of using the Galaxy S24 Ultra, I’m genuinely impressed with its performance—Snapdragon 8 Gen 3 delivers blazing-fast multitasking, and the 120Hz AMOLED display is stunning in both sunlight and dark modes. The AI-powered camera excels in low light, though edge detection in portrait mode can be inconsistent. Battery life comfortably lasts a day and a half with heavy use, but the phone does get noticeably warm during gaming. OneUI feels refined, though still cluttered compared to stock Android. Overall, it's a powerhouse with minor quirks that serious users might overlook for the sheer capability""")

for key, value in result.items():
    print(f"{key}: {value}")






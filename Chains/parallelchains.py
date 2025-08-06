from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel #is used to run 2 chain paralelly i.e invoke to two chains at same time 

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"]= os.getenv("google_api_key")

llm= HuggingFaceEndpoint(
    model= "openai/gpt-oss-120b"
)

model1= ChatHuggingFace(
    llm= llm
)

model2= GoogleGenerativeAI(
    model= "gemini-2.0-flash"
)

#prompt to generate notes of given text
prompt1= ChatPromptTemplate.from_template(
    """Generate a note/summary on the basis of given text
    <text>
    {text}
    </text>
    """
)

#prompt to generate a quiz of given text
prompt2= ChatPromptTemplate.from_template(
    """Generate a 5 quizes on the basis of given text that may be asked in interview/exams
    <text>
    {txt}
    </text>
    """
)

#prompt to merge quiz and note
prompt3= ChatPromptTemplate.from_template(
    """Now merge the given notes and quiz as same output
    notes -> {notes}
    quiz -> {quiz}
    """
)


parser= StrOutputParser() #returns parsed string output from llm

""" 
defining runnableparallel chain
the chains in runnableparallel gets invoked at the same time.
inside runnable parallel we pass mutiple chains in form of dictionary.
the name of key in dictionary should be same as the name of dynamic variables of prompt template where the output of this chain will be invoked
like : in parallel_chain the keys are notes, quiz which is same as the dynamic variable in prompt3.
The chains of the parallelchain should be independent i.e the output of one chain shouldnot be used as input of another chain
"""


parallel_chain= RunnableParallel(
    {
        "notes" : prompt1| model2| parser,  #chain to generate note of given text
        "quiz": prompt2| model1| parser    #chain to generate quiz of given text
    }
)

#chain to merge the outputs received from parallel_chains into single output
merge_chain= prompt3| model2| parser

final_chain= parallel_chain| merge_chain
text= """A black hole is one of the most fascinating and mysterious objects in the universe. At its core, a black hole is a region in space where gravity is so intense that nothing—not even light—can escape its pull. This immense gravitational force arises when a massive amount of matter is compressed into an extremely small area, typically as a result of a massive star collapsing under its own gravity at the end of its life cycle. The boundary surrounding a black hole is known as the *event horizon*. Once something crosses this boundary, it is inevitably pulled inward, with no possibility of return. Because light cannot escape, black holes are invisible, and their presence must be inferred by observing their effects on nearby matter and radiation.

Black holes come in different sizes and types. The most common classification includes **stellar-mass black holes**, **supermassive black holes**, and **intermediate black holes**. Stellar black holes are formed from the remnants of massive stars that have exhausted their nuclear fuel. They typically have masses between 5 to 100 times that of our Sun. On the other hand, supermassive black holes—millions to billions of times the mass of the Sun—reside at the centers of most galaxies, including the Milky Way. Their origins are still not fully understood, but they play a crucial role in galaxy formation and evolution. Intermediate black holes are thought to exist in between these two types in terms of mass, although their existence is harder to confirm due to observational limitations.

The **physics of black holes** is deeply tied to Einstein's theory of general relativity. According to this theory, a black hole distorts spacetime to an extreme degree. The core of a black hole, called the *singularity*, is a point where density becomes infinite, and the laws of physics as we know them break down. Time and space cease to have their usual meanings near this singularity. Surrounding the event horizon is a region where the gravitational pull warps both time and space significantly, creating phenomena such as gravitational time dilation—where time moves slower relative to outside observers.

Despite being invisible, black holes can be detected through indirect methods. When a black hole pulls in surrounding matter—like gas or stars—it can form an **accretion disk**. This material heats up due to friction and emits intense radiation, particularly in the X-ray spectrum, which astronomers can detect. Additionally, when two black holes merge, they release powerful **gravitational waves**, ripples in the fabric of spacetime, which have been observed by instruments such as LIGO and Virgo. These discoveries have opened a new era in astronomy, allowing scientists to study black holes not just through light, but through the very vibrations of space itself.

In recent years, advancements in technology and theory have allowed for remarkable achievements, such as the **first-ever image of a black hole’s event horizon**, captured by the Event Horizon Telescope in 2019. This historic image showed a glowing ring of gas surrounding the dark shadow of a supermassive black hole at the center of galaxy M87. This accomplishment was a major confirmation of theoretical predictions and provided visual evidence supporting the existence and nature of black holes.

Ultimately, black holes remain one of the most intriguing and enigmatic subjects in astrophysics. They challenge our understanding of physics, especially in the realms of gravity, time, and quantum mechanics. Ongoing research into black holes may eventually help scientists unlock deeper truths about the origin, structure, and fate of the universe itself.
"""

output= final_chain.invoke({"text": text,
                            "txt": text})
print(output)



#to show the flowchart of chains used in final_chain
final_chain.get_graph().print_ascii()



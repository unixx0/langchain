import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_kEY"]= os.getenv("google_api_key")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain.chains.query_constructor import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever, AttributeInfo
from langchain_experimental.text_splitter import SemanticChunker


html_string = """
<!DOCTYPE html>
  <html lang='en'>
  <head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Fancy Example HTML Page</title>
  </head>
  <body>
    <h1>Main Title</h1>
    <p>This is an introductory paragraph with some basic content.</p>
    
    <h2>Section 1: Introduction</h2>
    <p>This section introduces the topic. Below is a list:</p>
    <ul>
      <li>First item</li>
      <li>Second item</li>
      <li>Third item with <strong>bold text</strong> and <a href='#'>a link</a></li>
    </ul>
    
    <h3>Subsection 1.1: Details</h3>
    <p>This subsection provides additional details. Here's a table:</p>
    <table border='1'>
      <thead>
        <tr>
          <th>Header 1</th>
          <th>Header 2</th>
          <th>Header 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Row 1, Cell 1</td>
          <td>Row 1, Cell 2</td>
          <td>Row 1, Cell 3</td>
        </tr>
        <tr>
          <td>Row 2, Cell 1</td>
          <td>Row 2, Cell 2</td>
          <td>Row 2, Cell 3</td>
        </tr>
      </tbody>
    </table>
    
    <h2>Section 2: Media Content</h2>
    <p>This section contains an image and a video:</p>
      <img src='example_image_link.mp4' alt='Example Image'>
      <video controls width='250' src='example_video_link.mp4' type='video/mp4'>
      Your browser does not support the video tag.
    </video>

    <h2>Section 3: Code Example</h2>
    <p>This section contains a code block:</p>
    <pre><code data-lang="html">
    &lt;div&gt;
      &lt;p&gt;This is a paragraph inside a div.&lt;/p&gt;
    &lt;/div&gt;
    </code></pre>

    <h2>Conclusion</h2>
    <p>This is the conclusion of the document.</p>
  </body>
  </html>
"""

# using HTMLHeaderTextSplitter
headers_to_split_on= [
    ("h1", "header 1"),
    ("h2", "header 2"),
    ("h3", "header 3"),
]


splitter= HTMLHeaderTextSplitter(headers_to_split_on, return_each_element= False)
splitted_html= splitter.split_text(html_string)
print(splitted_html)



#to split directly from url
splitted_html= splitter.split_text_from_url("https://www.youtube.com/")
#print(splitted_html)

from langchain_text_splitters import HTMLSemanticPreservingSplitter
html_string = """
<!DOCTYPE html>
<html>
    <body>
        <div>
            <h1>Section 1</h1>
            <p>This section contains an important table and list that should not be split across chunks.</p>
            <table>
                <tr>
                    <th>Item</th>
                    <th>Quantity</th>
                    <th>Price</th>
                </tr>
                <tr>
                    <td>Apples</td>
                    <td>10</td>
                    <td>$1.00</td>
                </tr>
                <tr>
                    <td>Oranges</td>
                    <td>5</td>
                    <td>$0.50</td>
                </tr>
                <tr>
                    <td>Bananas</td>
                    <td>50</td>
                    <td>$1.50</td>
                </tr>
            </table>
            <h2>Subsection 1.1</h2>
            <p>Additional text in subsection 1.1 that is separated from the table and list.</p>
            <p>Here is a detailed list:</p>
            <ul>
                <li>Item 1: Description of item 1, which is quite detailed and important.</li>
                <li>Item 2: Description of item 2, which also contains significant information.</li>
                <li>Item 3: Description of item 3, another item that we don't want to split across chunks.</li>
            </ul>
        </div>
    </body>
</html>
"""

headers_to_split_on= [
    ("h1", "header 1"),
    ("h2", "header 2"),

]


html_splitter= HTMLSemanticPreservingSplitter(
    headers_to_split_on= headers_to_split_on,
    elements_to_preserve= ["table", "ul"],
    max_chunk_size= 100,
    preserve_images= True,    #includes content of <images> inside chunk
    denylist_tags= ["head", "style"]  #doesnt include the content of this inside chunks

)
splitted_html= html_splitter.split_text(html_string)
print(splitted_html)



#using a custom handler
def custom_iframe_extractor(iframe_tag):    
    iframe_src= iframe_tag.get("src", "")  #gets the src portion from iframe tag(.text to get the text contents of the tags)
    return f"[iframe: {iframe_src}]({iframe_src})"

splitter= HTMLSemanticPreservingSplitter(
    headers_to_split_on= headers_to_split_on,
    max_chunk_size= 60,
    elements_to_preserve= ["table", "ol"],
    custom_handlers= {"iframe": custom_iframe_extractor}   #calls iframe extractor function when chunker sees <iframe> tag
)




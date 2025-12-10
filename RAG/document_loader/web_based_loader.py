from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/canon-eos-r10-mirrorless-camera-body-rf-s-18-150-mm-f-3-5-6-3-stm-lens-kit/p/itmd3e8ee6e3f328?pid=DLLGFXCMZJJAQVG6&lid=LSTDLLGFXCMZJJAQVG60NKDKR&marketplace=FLIPKART&q=canon+r10&store=jek%2Fp31%2Ftrv&srno=s_1_1&otracker=search&otracker1=search&fm=search-autosuggest&iid=05ec0e02-ecdc-4b61-aff1-2e6a2a07d5e3.DLLGFXCMZJJAQVG6.SEARCH&ppt=sp&ppn=sp&ssid=cj8wzig1y80000001765356188863&qH=b0b726f37c6723d9'
loader = WebBaseLoader(url)

docs= loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'what is the MP and price of the camera ?','text':docs[0].page_content}))

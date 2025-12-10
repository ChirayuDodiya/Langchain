from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

prompt = PromptTemplate(
    template='Write a summary for following text - \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('Alice’s_Adventures_in_Wonderland.txt',encoding='utf-8')

docs = loader.load()

# print(type(docs))
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))
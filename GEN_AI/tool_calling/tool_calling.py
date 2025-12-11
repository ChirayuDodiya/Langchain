from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

#tool create
@tool
def multiply(a:int,b:int)->int:
    """given 2 numbers a and b this tool returns their product"""
    return a*b

#tool binding
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)
llm_with_tools=llm.bind_tools([multiply])

query = HumanMessage('can you multiply 25 with 40')
messages=[query]
result=llm_with_tools.invoke(messages)
messages.append(result)


#tool execution
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)
print(llm_with_tools.invoke(messages))
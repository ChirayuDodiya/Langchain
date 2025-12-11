from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
import os
from dotenv import load_dotenv

load_dotenv()

#tool create
from langchain_core.tools import InjectedToolArg
from typing import Annotated
@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    """
    this function fetches the currency conversion factor between a given base currency symbol and a target currency symbol
    """
    url = os.getenv("CURRENCY_EXCHANGE")
    url+=base_currency+"/"+target_currency

    responce = requests.get(url)

    return responce.json()

# print(get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'}))

@tool
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency_value * conversion_rate

# print(convert.invoke({'base_currency_value':10,'conversion_rate':89.9456}))

#tool binding
llm =  ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)
llm_with_tools = llm.bind_tools([get_conversion_factor,convert])


messages = [HumanMessage('what is conversion factor between USD and INR,and based on that can you convert 10 USD to INR')]

ai_message=llm_with_tools.invoke(messages)

messages.append(ai_message)

# print(ai_message.tool_calls)
import json
for tool_call in ai_message.tool_calls:
    # print(tool_call)
    # execute the 1st tool and get the value of conversion rate
    if tool_call['name']=='get_conversion_factor':
        tool_message1=get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name']=='convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

print(llm_with_tools.invoke(messages))
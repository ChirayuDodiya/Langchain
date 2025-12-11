from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_classic import hub

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

response = agent_executor.invoke({"input":"3 ways to reach goa from delhi"})

print(response)
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1,max_tokens=1000)

chatHistory=[
    SystemMessage(content="you are a helpful ai assistant")
]

while True:
    user_input=input("you:")
    chatHistory.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    else: 
        responce = model.invoke(chatHistory)
        chatHistory.append(AIMessage(content=responce.content))
        print("ai:",responce.content)

print(chatHistory)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ------------------- Load environment variables -------------------
load_dotenv()

# ------------------- Initialize model and parser -------------------
model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1,max_tokens=1000)
output_parser = StrOutputParser()

# ------------------- Requirement Statement -------------------
requirement_text = """
E- Post Office System is the shopping portal of the world renowned postal service on the internet. It sells
Stamps, Postcards, Packets, Cartons, and also has services like courier, registering for electricity vendors,
selling mobile cards, etc. Under this website many products and services can be ordered, that are also
available in a "normal" branch. The product prices are identical with the prices of their normal branches.
The system can serve many customers at a time. Each customer will be required to register on the portal
and should login into the system. The customer will then be able to browse the products and services
available. The customer will choose one or more products/services and will place an order for the selected
items. The customer must provide the account details for the payment. After verifying the payment
details, the system will finalize the order and will send the acknowledgement i.e. order details and status)
to the customer. The customer can view the order status at any time and can cancel the order before the
shipping has started
"""

# ------------------- Step 1: Actor Extraction -------------------
actor_extractor = ChatPromptTemplate.from_messages([
    ("system", "You are a requirements analyst. Extract the actors with short role descriptions."),
    ("human",
     "Problem: 'Online Food Delivery system where customers place orders and delivery staff deliver food.'\n"
     "Actors:\n- Customer — primary: places food orders\n- Delivery Staff — primary: delivers orders\n- Restaurant — secondary: prepares food\n- Payment Gateway — secondary: processes payments\n\n---\n"
     "Now extract actors for:\nProblem: {req}\nActors:")
])

actor_chain = actor_extractor | model | output_parser
actors_result = actor_chain.invoke({"req": requirement_text})

print("\n=== Actors ===\n")
print(actors_result)

# ------------------- Step 2: Goal Identification -------------------
goal_extractor = ChatPromptTemplate.from_messages([
    ("system", "From the problem and actor list, identify clear goals (functional objectives)."),
    ("human",
     "Example:\nActors: Customer, Restaurant\nGoals:\nG1: Customer — Place an order\nG2: Restaurant — Prepare ordered meal\n\n---\n"
     "Problem: {req}\nActors: {actors}\nGoals:")
])

goal_chain = goal_extractor | model | output_parser
goals_result = goal_chain.invoke({"req": requirement_text, "actors": actors_result})

print("\n=== Goals ===\n")
print(goals_result)

# ------------------- Step 3: Use Case Generation -------------------
usecase_generator = ChatPromptTemplate.from_messages([
    ("system", "Generate detailed use cases from each goal in structured format."),
    ("human",
     "Problem: {req}\nActors: {actors}\nGoals: {goals}\n"
     "Use Cases:")
])

usecase_chain = usecase_generator | model | output_parser
usecases_result = usecase_chain.invoke({
    "req": requirement_text,
    "actors": actors_result,
    "goals": goals_result
})

print("\n=== Use Cases ===\n")
print(usecases_result)

# ------------------- Step 4: Relationship Extraction -------------------
relation_extractor = ChatPromptTemplate.from_messages([
    ("system", "Extract associations, includes, extends, and generalizations between use cases."),
    ("human",
     "Use Cases: {usecases}\nRelations:")
])

relation_chain = relation_extractor | model | output_parser
relations_result = relation_chain.invoke({"usecases": usecases_result})

print("\n=== Relations ===\n")
print(relations_result)

# ------------------- Step 5: Generate PlantUML -------------------
uml_generator = ChatPromptTemplate.from_messages([
    ("system", "Generate PlantUML code only (between @startuml and @enduml)."),
    ("human",
     "Actors: {actors}\nUse Cases: {usecases}\nRelations: {relations}\n"
     "PlantUML:")
])

uml_chain = uml_generator | model | output_parser
uml_result = uml_chain.invoke({
    "actors": actors_result,
    "usecases": usecases_result,
    "relations": relations_result
})

print("\n=== PlantUML Code ===\n")
print(uml_result)

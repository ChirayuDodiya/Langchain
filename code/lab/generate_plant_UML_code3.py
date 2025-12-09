from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

payroll_problem = """Design a class diagram for a payroll system that handles different types of Employees. All
employees have a name and an ID, but their salary calculation methods vary. Full-Time
Employees receive a fixed monthly salary. Part-Time Employees are paid an hourly rate and their
salary depends on the hours they worked. Contractors get a fixed payment per project. The
system should be able to process payroll for any type of employee using a single, unified method,
showcasing polymorphism."""

ticket_distribution_problem="""Metro station wants to establish a TicketDistributor machine that issues tickets for
passengers travelling on metro rails. Travellers have options of selecting a ticket for a single
trip, round trips or multiple trips. They can also issue a metro pass for regular passengers or
a time card for a day, a week or a month according to their requirements. The discounts on
tickets will be provided to frequent travelling passengers. The machine is also supposed to
read the metro pass and time cards issued by the metro counters or machine. The ticket rates
differ based on whether the traveller is a child or an adult. The machine is also required to
recognize original as well as fake currency notes. The typical transaction consists of a user
using the display interface to select the type and quantity of tickets and then choosing a
payment method of either cash, credit/debit card or smartcard. The tickets are printed and
dispensed to the user. Also, the messaging facilities after every transaction are required on
the registered number. The system can also be operated comfortably by a touch-screen. A
large number of heavy components are to be used. We do not want our system to slow down,
and also the usability of the machine.
The TicketDistributor must be able to handle several exceptions, such as aborting the
transaction for incomplete transactions, the insufficient amount given by the travellers to the
machine, money return in case of an aborted transaction, change return after a successful
transaction, showing insufficient balance in the card, updated information printed on the
tickets e.g. departure time, date, time, price, valid from, valid till, validity duration, ticket
issued from and destination station. In case of exceptions, an error message is to be displayed.
We do not want user feedback after every development stage but after every two stages to
save time. The machine is required to work in a heavy load environment such that in the
morning and evening time on weekdays, and weekends performance and efficiency would
not be affected."""

###### Extract Actors Using problem statement

class_prompt = ChatPromptTemplate.from_messages([
("system", 
    "You are a senior software architect. Your job is to extract a full class model from a software requirement. Include:\n"
    "- Class names\n"
    "- Attributes for each class\n"
    "- Methods for each class\n"
    "- Relationships between classes\n"
    "- Relationship type (Association, Inheritance, Aggregation, Composition)\n"
    "- Direction and cardinality (1:1, 1:N, N:M, etc.)\n"
    "- Optional: A brief description of each class.\n"
    "Format the output in plain structured text.") , 

    ("user", """Problem Statement : Employee and Payroll System
                ==> Classes:
     
                    1. Student
                    -> Description: A person enrolled in the university.
                    -> Attributes: name, studentID
                    -> Methods: enroll(course), drop(course)

                    2. Professor
                    -> Description: A person who teaches courses.
                    -> Attributes: name, employeeID
                    -> Methods: assignCourse(course)

                    3. Course
                    -> Description: An academic unit offered by the university.
                    -> Attributes: name, code, schedule
                    -> Methods: addStudent(student), removeStudent(student)

                    4. University
                    -> Description: The organization managing students, professors, and courses.
                    -> Attributes: name, address
                    -> Methods: registerStudent(student), hireProfessor(professor)

                ==> Relationships:

                    -> Student --enrolls in--> Course (Association) [1:N]  
                    -> Professor --teaches--> Course (Association) [1:N]  
                    -> University --manages--> Student, Professor, Course (Aggregation) [1:N]
                
                Do NOT include or repeat the example classes above. Generate the class model only for the following problem:

                Now generate a class model for the following problem: {problem}

                """)

])

### Class chain for identification of class , relationship , cardinality #####

class_chain = class_prompt | llm | parser
# class_chain_out = class_chain.invoke({"problem": payroll_problem})
class_chain_out = class_chain.invoke({"problem": ticket_distribution_problem})
print("\n--- Class Diagram Requirements ---\n")
print(class_chain_out)


###### Chain for Convert class model text to PlantUML ######


uml_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert software engineer. Convert structured class diagram descriptions into PlantUML. "
     "Respond ONLY with valid PlantUML syntax including @startuml and @enduml."),

    ("user", """Here is the class diagram description:

    {class_description}

    Generate a valid PlantUML class diagram code with cardinaliy .""")
])

uml_chain = uml_prompt | llm | parser
uml_output = uml_chain.invoke({"class_description": class_chain_out})

print("\n=== PlantUML Output ===\n")
print(uml_output)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
output_parser = StrOutputParser()

requirement_text="""
The hospital needs a system to manage patients, doctors, and appointments efficiently.
There are two types of doctors: general physicians for regular checkups and specialists for
specific areas like cardiology or orthopedics. Patients can register with their details and
medical history, and appointments can be booked based on doctor availability, with
emergency cases given priority automatically. The system will also include billing to calculate
consultation fees, test charges, and other costs.
"""

requirement_text2="""
The library intends to implement a new system to effectively manage books and their
circulation among its members. The system will handle different categories of users,
primarily students and faculty members, each with their own borrowing limits. Students can
borrow up to three books at a time, while faculty members can borrow up to ten books
simultaneously.Each book in the library will have specific attributes such as title, author,
ISBN number, and availability status. The library will be managed by a librarian, who will be
responsible for maintaining the collection by adding, updating, or removing books from the
system.
Members will be able to perform essential operations such as borrowing, returning, and
reserving books. The system must also keep track of due dates and calculate late fees for
overdue books based on the number of days they are returned late. This will help ensure
that books are circulated efficiently and made available to other members in a timely
manner.The system should be capable of handling various exceptions. For example, when a
member attempts to borrow a book that is already checked out, when they try to borrow
more books than their permitted limit, or when a return is attempted for a book that has no
active loan record in the system. In all such cases, the system must display appropriate error
messages to guide the user.
Additionally, the system must be able to generate monthly reports to provide insights into
library operations. These reports will include details such as the most borrowed books,
pending fines, and an overview of the library’s overall circulation activity. This information will
assist the librarian in making informed decisions about resource allocation, identifying
popular books for future purchases, and monitoring overdue fines.
The goal of this Library Management System is to ensure a seamless and efficient process
for borrowing and managing books, thereby enhancing the experience for both library
members and the librarian while maintaining accuracy and reliability
"""

class_data_extractor = ChatPromptTemplate.from_messages([
    ("system", "You are a senior software engineer.give all required classes,relation between classes,attribute and methods in classes and cardinality for class diagram"),
    ("human",
     "Problem: 'a univercity course registration systen where we need to store all student data with their corusponding field and branches'"
     "Answer"
     "classes: student \n branch \n field \n\n  relations - student - branch \n branch - field \n\n attributes: \n student: name id branch batch pointer \n branch: branch_name capacity field \n fields: field_name starting_year \n\n methods: student:check_pointer \n branch: add_branch remove_branch check_capacity get_student_list \n\n cardinality: student 0..* branch 1 \n branch 0..* fields 1\n"
     "Now extract class diagram information like above for:\n problem:{req}\nAnswer:")
])

class_chain = class_data_extractor | model | output_parser
class_result = class_chain.invoke({"req":requirement_text})

print("\n=== Answer ===\n")
print(class_result)

uml_generator = ChatPromptTemplate.from_messages([
    ("system", "Generate PlantUML code only (between @startuml and @enduml)."),
    ("human",
     "Problem: {req}\nAnswer:{ans}"
     "PlantUML:")
])


uml_chain = uml_generator | model | output_parser
uml_result = uml_chain.invoke({
    "req":requirement_text,"ans": class_result
})


print("\n=== PlantUML Code ===\n")
print(uml_result)
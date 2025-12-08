from pydantic import BaseModel,EmailStr,Field
#validates data type at run-time
from typing import Optional
class Student(BaseModel):
    name: str='defaultName'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10,default=5,description='A decimal value representing the cgpa of the student')

# new_student = {}
new_student = {'name':'chirayu','email':'abc@gmail.com','cgpa':7.18}

student = Student(**new_student)

print(student)
print(student.name)

student_dict = dict(student)
print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)
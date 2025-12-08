from typing import TypedDict
# doesn't validate data type at run-time
class Person(TypedDict):
    name:str
    age:int

new_person: Person = {'name':'chirayu','age':'19'}

print(new_person)
print(new_person['age'])
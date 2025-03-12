from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Person(BaseModel):

    # name : str
    name : str = "ajay"
    age : Optional[int] = None
    email : EmailStr
    marks : int = Field(gt=70, lt=90, default=45, description="The marks of the person in college")

# new_person = {"name" : "ajay"}
# new_person1 = {"name" : 26}
# new_person = {"age" : 26}
# new_person = {"age" : "26", "email" : "abc@gmail.com", "marks" : 80} # type coerceing
new_person = {"age" : "26", "email" : "abc@gmail.com"}

person = Person(**new_person)
# person1 = Person(**new_person1)

print(person)
# print(type(person))
# print(person1)

person_dict = dict(person)
print(person_dict["marks"])

person_json = person.model_dump_json()
print(person_json)
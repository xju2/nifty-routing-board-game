from person_pb2 import Person

person = Person()

with open("person.bin", "rb") as f:
    person.ParseFromString(f.read())

print("Name:", person.name)
print("ID:", person.id)

for email in person.email:
    print("Email:", email)

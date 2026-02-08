#include <fstream>
#include "person.pb.h"

int main() {
    example::Person person;

    person.set_name("Alice");
    person.set_id(42);
    person.add_email("alice@lbl.gov");
    person.add_email("alice@example.com");

    // Serialize to binary file
    std::ofstream out("person.bin", std::ios::binary);
    person.SerializeToOstream(&out);

    return 0;
}

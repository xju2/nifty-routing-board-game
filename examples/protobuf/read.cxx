#include <fstream>
#include <iostream>
#include "person.pb.h"

int main() {
    example::Person person;

    std::ifstream in("person.bin", std::ios::binary);
    person.ParseFromIstream(&in);

    std::cout << "Name: " << person.name() << "\n";
    std::cout << "ID: " << person.id() << "\n";

    for (const auto& email : person.email()) {
        std::cout << "Email: " << email << "\n";
    }

    return 0;
}

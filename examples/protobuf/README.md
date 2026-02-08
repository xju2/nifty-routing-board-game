## Instrucitons
Compile the `protobuf` example:
```bash
make
```

Then create a binary file for a person: `./create_person` and read it `./read_person`.
You can also read it from Python `python read.py`.


### Backup.

```bash
protoc --cpp_out=. --python_out=. person.proto

clang++ create.cxx person.pb.cc $(pkg-config --cflags --libs protobuf) -pthread -std=c++17 -o create_person

clang++ read.cxx person.pb.cc $(pkg-config --cflags --libs protobuf) -pthread -std=c++17 -o read_person
```
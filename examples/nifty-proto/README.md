## Instructions
1. Run `make` from this directory to generate `nifty_common.pb.*` and build the `create.exe` and `read.exe` binaries.
2. `./create.exe` serializes a mock `GraphBatch` (100 nodes, 3 features, 3 edges) and writes it to `nifty_common.bin`.
3. `./read.exe` parses `nifty_common.bin` and prints version, creator, node/edge counts.
4. `python read.py` does the same using the Python protobuf API.

### Tips
- `make proto` only regenerates the protobuf sources.
- `make clean` removes object/binary artifacts; `make distclean` also removes the generated `.pb.*` files and the Python stub.

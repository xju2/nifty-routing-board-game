CLANG ?= clang
TARGET ?= wasm32
WASM=main.wasm

all: $(WASM)

$(WASM): main.c
	$(CLANG) --target=$(TARGET) -O3 -nostdlib -ffreestanding -fno-builtin \
	  -fno-builtin-memcpy -fno-builtin-memmove -fno-builtin-memset -fno-builtin-memcmp -fno-builtin-bzero \
	  -fuse-ld=lld \
	  -Wl,--no-entry \
	  -Wl,--export=init -Wl,--export=frame -Wl,--export=set_viewport \
	  -Wl,--export=on_pointer -Wl,--export=on_key \
	  -Wl,--export-memory \
	  -Wl,--initial-memory=2097152 -Wl,--max-memory=16777216 \
	  -Wl,--export-table \
	  -Wl,--allow-undefined \
	  -o $(WASM) main.c

.PHONY: clean
clean:
	rm -f $(WASM)

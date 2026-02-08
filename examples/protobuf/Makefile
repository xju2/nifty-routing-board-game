# Portable Makefile for Linux and macOS
# - Generates person.pb.cc / person.pb.h via protoc
# - Builds create_person and read_person
# - Uses pkg-config when available, but allows manual override
#
# Usage:
#   make                # build both programs
#   make create_person  # build create_person
#   make read_person    # build read_person
#   make proto          # generate pb sources only
#   make clean
#   make distclean      # clean + remove generated proto files
#   make help

SHELL := /bin/sh

# ---------- Toolchain selection (can override from env) ----------
CXX ?= $(shell command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null || echo g++)
PROTOC ?= $(shell command -v protoc 2>/dev/null || echo protoc)
PKGCONFIG := $(shell command -v pkg-config 2>/dev/null || true)

# ---------- Basic flags ----------
CXXFLAGS  := -std=c++17 -Wall -Wextra
LDFLAGS   :=

# Detect platform
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    IS_DARWIN := 1
else
    IS_DARWIN := 0
endif

# If using clang++ on macOS, prefer libc++
ifeq ($(findstring clang,$(CXX)),clang)
  ifeq ($(IS_DARWIN),1)
    CXXFLAGS += -stdlib=libc++
  endif
endif

# Allow user to override pkg-config result by setting these env vars:
#    PROTOBUF_CFLAGS, PROTOBUF_LIBS
ifdef PKGCONFIG
  ifneq ($(shell $(PKGCONFIG) --exists protobuf && echo yes),)
    PROTOBUF_CFLAGS ?= $(shell $(PKGCONFIG) --cflags protobuf)
    PROTOBUF_LIBS   ?= $(shell $(PKGCONFIG) --libs protobuf)
  endif
endif

# If pkg-config didn't populate PROTOBUF_* variables, user can set them externally.
# Fallback: empty, which will compile but may fail linking if libs not provided.
PROTOBUF_CFLAGS ?=
PROTOBUF_LIBS   ?=

# If pthread needed explicitly on some systems
LDFLAGS += $(PROTOBUF_LIBS)
CXXFLAGS += $(PROTOBUF_CFLAGS)

# ---------- Sources and build layout ----------
BUILD_DIR ?= .
SRCS_CXX := create.cxx read.cxx
PROTO    := person.proto
PB_CC    := $(BUILD_DIR)/person.pb.cc
PB_H     := $(BUILD_DIR)/person.pb.h

OBJS := $(patsubst %.cxx,%.o,$(SRCS_CXX))
OBJS := $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
OBJS += $(BUILD_DIR)/person.pb.o

TARGETS := create_person read_person

.PHONY: all proto clean distclean help

all: $(TARGETS)

# Link targets
create_person: $(BUILD_DIR)/create.o $(BUILD_DIR)/person.pb.o
	$(CXX) $^ $(LDFLAGS) -o $@

read_person: $(BUILD_DIR)/read.o $(BUILD_DIR)/person.pb.o
	$(CXX) $^ $(LDFLAGS) -o $@

# Compile rules (put .o files into BUILD_DIR)
$(BUILD_DIR)/%.o: %.cxx $(PB_CC) $(PB_H) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/person.pb.o: $(PB_CC) $(PB_H) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(PB_CC) -o $@

# Protobuf generation: write files to BUILD_DIR
# protoc --cpp_out=DIR person.proto
$(PB_CC) $(PB_H): $(PROTO) | $(BUILD_DIR)
	@echo "Generating protobuf sources from $< -> $(BUILD_DIR)"
	$(PROTOC) --cpp_out=$(BUILD_DIR) --python_out=$(BUILD_DIR) $(PROTO)

proto: $(PB_CC) $(PB_H)

# Ensure build dir exists (no-op if BUILD_DIR=".")
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	@echo "Cleaning object files and targets..."
	@rm -f $(BUILD_DIR)/*.o $(TARGETS)
	@rm -f $(BUILD_DIR)/*.pb.*
	@rm -f $(BUILD_DIR)/*pb2.py

distclean: clean
	@echo "Removing generated protobuf sources..."
	@rm -f $(PB_CC) $(PB_H)

help:
	@printf "Makefile targets:\n"
	@printf "  make         Build all targets (create_person, read_person)\n"
	@printf "  make create_person\n"
	@printf "  make read_person\n"
	@printf "  make proto    Generate $(PB_CC) and $(PB_H) from $(PROTO)\n"
	@printf "  make clean    Remove object files and binaries\n"
	@printf "  make distclean  clean + remove generated proto files\n\n"
	@printf "Environment overrides:\n"
	@printf "  CXX         (default: autodetect clang++ then g++)\n"
	@printf "  PROTOC      (default: autodetect protoc)\n"
	@printf "  BUILD_DIR   (default: .) — change output dir for objects & generated pb files\n"
	@printf "  PROTOBUF_CFLAGS / PROTOBUF_LIBS  (if pkg-config is missing)\n\n"
	@printf "Examples:\n"
	@printf "  make\n"
	@printf "  make BUILD_DIR=build       # out-of-tree build\n"
	@printf "  make PROTOBUF_LIBS='-L/opt/homebrew/lib -lprotobuf -labsl_log' PROTOBUF_CFLAGS='-I/opt/homebrew/include'\n"

# Print a short diagnostic if pkg-config didn't supply protobuf flags
# This target runs automatically if you call 'make help' or others won't warn; run 'make pkgcheck' for explicit check.
.PHONY: pkgcheck
pkgcheck:
ifeq ($(strip $(PROTOBUF_CFLAGS)$(PROTOBUF_LIBS)),)
	@echo "WARNING: pkg-config did not provide protobuf flags and PROTOBUF_* variables are empty."
	@echo "If you get link/compile errors, set PROTOBUF_CFLAGS and PROTOBUF_LIBS in the environment."
	@echo "Example (macOS Homebrew):"
	@echo "  make PROTOBUF_CFLAGS='-I/opt/homebrew/include' PROTOBUF_LIBS='-L/opt/homebrew/lib -lprotobuf -labsl_log'"
else
	@echo "PROTOBUF flags detected:"
	@echo "  PROTOBUF_CFLAGS = $(PROTOBUF_CFLAGS)"
	@echo "  PROTOBUF_LIBS   = $(PROTOBUF_LIBS)"
endif

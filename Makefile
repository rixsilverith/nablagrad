CC := g++ -std=c++17
CFLAGS := -g -Wall -O2
LDFLAGS := -Lbuild -lnablagrad

BUILD_DIR := build
INSTALL_DIR := /usr/include
INSTALL_LIB_DIR := /usr/lib
NABLA_DIR := nablagrad
SRCS := $(NABLA_DIR)/tensor.cpp
# SRCS := $(NABLA_DIR)/dual.cpp $(NABLA_DIR)/tensor.cpp $(NABLA_DIR)/core.cpp $(NABLA_DIR)/forward_ad.cpp $(NABLA_DIR)/gradient_tape.cpp $(NABLA_DIR)/tensor_ops.cpp
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
EXAMPLES_DIR := examples
EXAMPLES := $(EXAMPLES_DIR)/reverse_mode_gradient $(EXAMPLES_DIR)/forward_mode_partial_diff

LIBRARY := libnablagrad.a

.PHONY: all build examples install uninstall clean

all: test

build: $(BUILD_DIR)/$(LIBRARY)

$(BUILD_DIR)/$(LIBRARY): $(OBJS)
	@if [ ! -d "build" ]; then mkdir "build"; fi
	ar rcs $@ $^

$(NABLA_DIR)/%.o: $(SRCS)
	$(CC) $(CFLAGS) -c $(NABLA_DIR)/$*.cpp -o $(NABLA_DIR)/$*.o

examples: $(EXAMPLES)

$(EXAMPLES_DIR)/%.o: $(EXAMPLES_DIR)/$*.cpp
	$(CC) $(CFLAGS) -c $(EXAMPLES_DIR)/$*.cpp -o $(EXAMPLES_DIR)/$*.o

$(EXAMPLES_DIR)/reverse_mode_gradient: $(EXAMPLES_DIR)/reverse_mode_gradient.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(EXAMPLES_DIR)/forward_mode_partial_diff: $(EXAMPLES_DIR)/forward_mode_partial_diff.o
	$(CC) -o $@ $^ $(LDFLAGS)

test: $(NABLA_DIR)/main.o $(OBJS)
	$(CC) -o $@ $(NABLA_DIR)/main.o $(OBJS)

install: build
	@cp $(BUILD_DIR)/$(LIBRARY) $(INSTALL_LIB_DIR)/$(LIBRARY)
	@cp -r $(NABLA_DIR) $(INSTALL_DIR)
	@echo "nablagrad installed to "$(INSTALL_DIR)" and "$(INSTALL_LIB_DIR)""

uninstall:
	@rm $(INSTALL_LIB_DIR)/$(LIBRARY)
	@rm -r $(INSTALL_DIR)/$(NABLA_DIR)
	@echo "nablagrad uninstalled"

clean:
	rm -f $(NABLA_DIR)/*.o $(EXAMPLES_DIR)/*.o $(BUILD_DIR)/$(LIBRARY) $(EXAMPLES) test

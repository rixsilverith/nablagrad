CC := g++ -std=c++17
CFLAGS := -g -Wall -O2

INSTALL_DIR := /usr/include
NABLA_DIR := nablagrad
SRCS := $(NABLA_DIR)/dual.cpp $(NABLA_DIR)/tensor.cpp $(NABLA_DIR)/core.cpp $(NABLA_DIR)/forward_ad.cpp $(NABLA_DIR)/gradient_tape.cpp $(NABLA_DIR)/tensor_ops.cpp
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all install uninstall clean

all: test

test: $(NABLA_DIR)/main.o $(OBJS)
	$(CC) -o $@ $(NABLA_DIR)/main.o $(OBJS)

$(NABLA_DIR)/%.o: $(SRCS)
	$(CC) $(CFLAGS) -c $(NABLA_DIR)/$*.cpp -o $(NABLA_DIR)/$*.o

install:
	cp -r $(NABLA_DIR) $(INSTALL_DIR)
	@echo "nablagrad installed to "$(INSTALL_DIR)""

uninstall:
	rm -r $(INSTALL_DIR)/$(NABLA_DIR)
	@echo "nablagrad uninstalled"

clean:
	rm $(NABLA_DIR)/*.o test

CC := g++ -std=c++17
CFLAGS := -g -Wall -O2

INSTALL_DIR := /usr/include

.PHONY: all install uninstall clean

all: test

test: nablagrad/main.o nablagrad/tensor.o nablagrad/core.o nablagrad/dual.o nablagrad/forward_ad.o 
	$(CC) -o $@ nablagrad/main.o nablagrad/tensor.o nablagrad/core.o nablagrad/dual.o nablagrad/forward_ad.o

nablagrad/main.o: nablagrad/main.cpp
	$(CC) $(CFLAGS) -c nablagrad/main.cpp -o nablagrad/main.o

nablagrad/forward_ad.o: nablagrad/forward_ad.cpp
	$(CC) $(CFLAGS) -c nablagrad/forward_ad.cpp -o nablagrad/forward_ad.o
nablagrad/core.o: nablagrad/core.cpp
	$(CC) $(CFLAGS) -c nablagrad/core.cpp -o nablagrad/core.o
nablagrad/dual.o: nablagrad/dual.cpp
	$(CC) $(CFLAGS) -c nablagrad/dual.cpp -o nablagrad/dual.o
nablagrad/tensor.o: nablagrad/tensor.cpp
	$(CC) $(CFLAGS) -c nablagrad/tensor.cpp -o nablagrad/tensor.o

install:
	cp -r nablagrad $(INSTALL_DIR)
	@echo "nablagrad installed to "$(INSTALL_DIR)""

uninstall:
	rm -r $(INSTALL_DIR)/nablagrad
	@echo "nablagrad uninstalled"

clean:
	rm nablagrad/*.o test

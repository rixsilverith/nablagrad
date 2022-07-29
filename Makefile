CC := g++
CFLAGS := -g -Wall -O2

.PHONY: all clean

all: test

test: lib/main.o lib/nablagrad.o lib/duals.o lib/forward_ad.o
	$(CC) -o $@ lib/main.o lib/nablagrad.o lib/duals.o lib/forward_ad.o

lib/main.o: lib/main.cpp
	$(CC) $(CFLAGS) -c lib/main.cpp -o lib/main.o

lib/forward_ad.o: lib/forward_ad.cpp
	$(CC) $(CFLAGS) -c lib/forward_ad.cpp -o lib/forward_ad.o
lib/nablagrad.o: lib/nablagrad.cpp
	$(CC) $(CFLAGS) -c lib/nablagrad.cpp -o lib/nablagrad.o
lib/duals.o: lib/duals.cpp
	$(CC) $(CFLAGS) -c lib/duals.cpp -o lib/duals.o

clean:
	rm lib/*.o test

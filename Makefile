CC := g++
CFLAGS := -g -Wall -O2

.PHONY: all clean

all: test

test: lib/main.o lib/autograd.o lib/dual.o lib/forward_ad.o
	$(CC) -o $@ lib/main.o lib/autograd.o lib/dual.o lib/forward_ad.o

lib/main.o: lib/main.cpp
	$(CC) $(CFLAGS) -c lib/main.cpp -o lib/main.o

lib/forward_ad.o: lib/forward_ad.cpp
	$(CC) $(CFLAGS) -c lib/forward_ad.cpp -o lib/forward_ad.o
lib/autograd.o: lib/autograd.cpp
	$(CC) $(CFLAGS) -c lib/autograd.cpp -o lib/autograd.o
lib/dual.o: lib/dual.cpp
	$(CC) $(CFLAGS) -c lib/dual.cpp -o lib/dual.o

clean:
	rm lib/*.o test

CC=c99

all: tests

clean:
	rm -f *.o tests/*.o tests/projection
tests: projection.o tests/projection.o
	$(CC) projection.o tests/projection.o -o tests/projection

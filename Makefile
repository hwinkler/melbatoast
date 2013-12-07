CC=c99

all: tests potential

clean:
	rm -f *.o tests/*.o tests/projection

tests: projection.o tests/projection.o
	$(CC) projection.o tests/projection.o -o tests/projection

projection.o: projection.c projection.h

potential.o: potential.c potential.h projection.h

potential: potential.o projection.o

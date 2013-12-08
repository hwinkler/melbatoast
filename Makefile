CC=c99
CFLAGS=-g  

all: tests potential

clean:
	rm -f potential  *.o  mtwist-1.4/*.o tests/*.o tests/projection

tests: projection.o tests/projection.o
	$(CC) projection.o tests/projection.o -o tests/projection

projection.o: projection.c projection.h

potential.o: potential.c potential.h projection.h

potential: potential.o projection.o mtwist-1.4/mtwist.o

mtwist-1.4/mtwist.o: mtwist-1.4/mtwist.c mtwist-1.4/mtwist.h

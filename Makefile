CC=c99
CFLAGS=-O3  -DNDEBUG

all: tests 

clean:
	rm -f   *.o  mtwist-1.4/*.o tests/*.o tests/jensen tests/projection

tests: tests/projection tests/jensen

tests/projection: projection.o tests/projection.o

tests/jensen: gibbs.o potential.o projection.o mtwist-1.4/mtwist.o tests/jensen.o

tests/projection.o: tests/projection.c

projection.o: projection.c projection.h

potential.o: potential.c potential.h

gibbs.o: gibbs.c gibbs.h potential.h projection.h

mtwist-1.4/mtwist.o: mtwist-1.4/mtwist.c mtwist-1.4/mtwist.h

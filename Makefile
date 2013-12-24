all: jensen

jensen: main.cu  gram.c parse.o potential.h cudacall.h 
	nvcc -arch=sm_20 main.cu gram.c parse.o -o jensen

parse.o: parse.c parse.h
	c99 -c parse.c

gram.c: gram.y parse.h
	lemon gram.y

clean:
	rm -f *.o  tests/*.o jensen main

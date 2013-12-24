all: jensen

jensen: main.cu  gram.c main.o parse.o gibbs.o 
	nvcc -arch=sm_20 main.o gram.c parse.o gibbs.o -o jensen

parse.o: parse.cu parse.h
	nvcc -c parse.cu

main.o: main.cu parse.h gibbs.h potential.h cudacall.h
	nvcc -c main.cu

gram.c: gram.y parse.h
	lemon gram.y

clean:
	rm -f *.o  tests/*.o jensen main

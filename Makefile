all: jensen

jensen: main.cu  gram.c parse.c potential.h cudacall.h 
	nvcc -arch=sm_20 main.cu gram.c parse.c -o jensen

gram.c: gram.y parse.h
	lemon gram.y

clean:
	rm -f *.o  tests/*.o jensen main

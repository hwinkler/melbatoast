CC=nvcc
CFLAGS=

GENCODE_FLAGS   := -arch=sm_20


all: jensen

jensen: main.cu  potential.h cudacall.h 
	nvcc -arch=sm_20 main.cu -o jensen
clean:
	rm -f         *.o  tests/*.o jensen main

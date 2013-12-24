all: jensen

jensen: main.o gram.o parse.o device.o gibbs.o 
	nvcc -arch=sm_35 main.o device.o gram.o parse.o gibbs.o -o jensen

gibbs.o: gibbs.cu gibbs.h potential.h
	nvcc -arch=sm_35 -dc gibbs.cu

parse.o: parse.cu parse.h
	nvcc -arch=sm_35 -c parse.cu

device.o: device.cu device.h cudacall.h
	nvcc -arch=sm_35  -c device.cu

gram.o: gram.cu parse.h
	nvcc -arch=sm_35 -c gram.cu

main.o: main.cu parse.h gibbs.h potential.h device.h cudacall.h
	nvcc -arch=sm_35 -c main.cu

gram.cu: gram.y parse.h
	lemon gram.y
	mv gram.c gram.cu

clean:
	rm -f *.o gram.cu gram.c tests/*.o jensen main

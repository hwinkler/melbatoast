

all: build

build: main

main: main.cu add.cu
	nvcc -arch=sm_30 -o $@ $+ $(LIBRARIES)

clean:
	rm -f   main   *.a   *.o  tests/*.o 

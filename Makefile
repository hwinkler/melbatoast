CC=nvcc
CFLAGS=

GENCODE_FLAGS   := -arch=sm_35


all: build

build: jensenDevLib.a jensen

jensen: jensen.o potential.o jensenLink.o jensenDevLib.a
	nvcc -o $@ $+ $(LIBRARIES)
jensen.o: jensen.cu potential.h cudacall.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
potential.o: potential.cu potential.h 
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
jensenDevLib.a:potential.o
	nvcc -lib -o $@ potential.o
jensenLink.o: jensen.o jensenDevLib.a
	nvcc -dlink $(GENCODE_FLAGS) -o $@ $^

clean:
	rm -f      *.a   *.o  tests/*.o jensen

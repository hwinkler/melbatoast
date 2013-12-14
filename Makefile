CC=nvcc
CFLAGS=

GENCODE_FLAGS   := -arch=sm_35


all: build

build: jensenDevLib.a jensen

jensen: jensen.o potential.o jensenLink.o jensenDevLib.a
	nvcc -o $@ $+ $(LIBRARIES)
jensen.o: jensen.cu gibbs.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
gibbs.o: gibbs.cu gibbs.h projection.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
projection.o: projection.cu
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
rnd.o: rnd.cu
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
potential.o: potential.cu
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
jensenDevLib.a: gibbs.o projection.o rnd.o
	nvcc -lib -o $@  gibbs.o projection.o rnd.o
jensenLink.o: jensen.o potential.o jensenDevLib.a
	nvcc -dlink $(GENCODE_FLAGS) -o $@ $^

clean:
	rm -f         *.o  tests/*.o jensen

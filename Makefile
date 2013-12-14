CC=nvcc
CFLAGS=

GENCODE_FLAGS   := -arch=sm_20


all: build

build: jensenDevLib.a jensen

jensen: jensen.o potential.o jensenLink.o jensenDevLib.a
	nvcc -o $@ $+ $(LIBRARIES)
jensen.o: jensen.cu gibbs.h potential.h cudacall.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
gibbs.o: gibbs.cu gibbs.h projection.h potential.h rnd.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
projection.o: projection.cu projection.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
rnd.o: rnd.cu rnd.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
potential.o: potential.cu potential.h cudacall.h
	nvcc  $(CFLAGS) -dc $(GENCODE_FLAGS) -o $@ -c $<
jensenDevLib.a: gibbs.o projection.o rnd.o
	nvcc -lib -o $@  gibbs.o projection.o rnd.o
jensenLink.o: jensen.o potential.o jensenDevLib.a
	nvcc -dlink $(GENCODE_FLAGS) -o $@ $^

clean:
	rm -f         *.o  tests/*.o jensen

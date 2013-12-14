#include <cuda.h>
#include <curand_kernel.h>

__device__ double rnd(curandState* globalState);
__device__ void rndSeed ( curandState * state );

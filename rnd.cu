#include <cuda.h>
#include <curand_kernel.h>

__device__ double rnd(curandState* state) {
  return curand_uniform( state );
}

__device__ void rndSeed ( curandState * state){
  unsigned int seed = (unsigned int) clock64();
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init ( seed, id, 0, &state[id] );
} 

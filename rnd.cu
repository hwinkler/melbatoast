#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__device__ double mt_drand(curandState* globalState) {
  return 0.34;
  //int ind = threadIdx.x;
  //curandState localState = globalState[ind];
  //return curand_uniform( &localState );
}
__device__  void mt_seed(){

}

__global__ void setup_rnd_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

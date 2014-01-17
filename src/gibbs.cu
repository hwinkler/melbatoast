#include <curand_kernel.h>
#include <assert.h>
#include "constants.h"
#include "potential.h"

#define DPRINT(...)

__device__ float rnd(curandState* state) {
  return curand_uniform( state );
}

__device__ void rndSeed ( curandState * state){
  unsigned int seed = (unsigned int) clock();
  int id =  threadIdx.x + blockIdx.x * blockDim.x;
  curand_init ( seed, id, 0, state );
} 


__device__
void projection (
                 const int *const dimensions,
                 const int *const indices,
                 const int numDimensions,
                 int *offsetOut,
                 int *lengthOut,
                 int *strideOut) {

  // the first dimension varies fastest
  // index -1 indicates the dimension we are extracting

  int hypersheetSize = 1;
  int stride = -1;
  int index0 = 0;
  int length = -1;
  
  for (int i =0; i < numDimensions ; i++){

    int index;
    if (indices[i] < 0){
      index = 0;
      stride = hypersheetSize;
      length = dimensions[i];
    } else {
      index = indices[i];
    }
    index0 += hypersheetSize * index;
    hypersheetSize *= dimensions[i];
  }

  *offsetOut = index0;
  *lengthOut = length;
  *strideOut = stride;

}


__device__
void _initDistribution (float* distribution, int n){
  #pragma unroll
  for (int i=0; i< n; i++){
    distribution[i] = 1.0f;
  }
}

__device__
void _normalizeDistribution ( float*  distribution, int n){
  float sum = 0.f;
  #pragma unroll
  for (int i=0; i< n; i++){
    assert(distribution[i] > 0.f);
    sum += distribution[i] ;
  }
  float scale = 1.f / sum;
  #pragma unroll
  for (int i=0; i< n; i++){
    distribution[i] *= scale ;
  }
}

__device__
void _cumulativeDistribution (const float* const distribution, float * cumulative, int n){
  cumulative[0] = distribution[0];
  for (int i=1; i< n; i++){
    cumulative[i] = cumulative[i-1] + distribution[i] ;
  }
}

__device__
int _drawFromCumulative (const float* const cumulative, int n, curandState* rndState){
  float r =  rnd(rndState);
  #pragma unroll
  for (int i=0; i<n; i++){
   
    if ( r <= cumulative[i]){
      return i;
    }
  }
  //DPRINT("drawFromCumulative %d %f %f\n", n, r, cumulative[n-1]);
  //assert (false);
  return n-1;
}

__device__
void _conditionalGiven(
    const Potential * __restrict potentials ,
    int offset,
    const int * __restrict states,
    int indexUnfixed, float* distribution){

  const Potential *const potential = potentials + offset;
  int numDimensions = potential->numParents + 1;
  int indices[MAX_DIMENSIONS];

  indices[0] = states[offset];

  #pragma unroll
  for (int iParent=0; iParent < potential->numParents; iParent++){
    assert (iParent+1 < numDimensions);
    Potential *parent = potential->parents[iParent];
    int parentOffset = parent - potentials;
    indices[iParent+1] = states[parentOffset];
  }

  assert (indexUnfixed < numDimensions);
  indices[indexUnfixed] = -1;

  int offsetOut = 0, lengthOut =0, strideOut =0;
  projection ( potential->dimensions, indices, numDimensions,
               &offsetOut, &lengthOut, &strideOut);
  
  for (int iState =0; iState < potential->numStates; iState++){
    assert (offsetOut + iState * strideOut < potential->numConditionals);
    distribution[iState] *= __ldg( potential->conditionals + offsetOut + iState * strideOut);
  }
}

__global__
void gibbs ( const Potential* __restrict  potentials, int numPotentials, const int *__restrict initialStates,
            int countsBase[], int numCounts, int numIterations) {
  curandState rndState;
  rndSeed(&rndState);

  if (numPotentials > MAX_POTENTIALS){
    return;
  }
  int states[MAX_POTENTIALS]; 
  memcpy (states, initialStates, numPotentials*sizeof(int));
  
//  int* counts = (int*) malloc(numCounts*sizeof(int));
//  memset ( counts, 0, numCounts* sizeof(int));

  for (int i=0; i<numIterations; i++){
    
    for (int j=0; j < numPotentials; j++){
      float distribution [MAX_STATES];
      float cumulative [MAX_STATES];

      const Potential *const p = potentials + j;
      if (p->isFrozen){
        continue;
      }
      _initDistribution(distribution, p->numStates);
     // __syncthreads();
      assert(distribution[0] == 1.0f);
           
      // Obtain the conditional distribution for the current potential

      _conditionalGiven (potentials, j, states, 0, distribution);     
      
      // Multiply in the distribution for this variable in each child potential
      for (int iChild =0; iChild < p->numChildren; iChild++){
        Potential * child = p->children[iChild];
     
        // add one to indexInChild, since the indexInChild refers to zero based
        // among the parents -- but zer oindex is reserved to the potential's variable
        _conditionalGiven (potentials, child - potentials, states, p->indexInChild[iChild] + 1, distribution);
        
      }
      
      _normalizeDistribution (distribution, p->numStates);

      _cumulativeDistribution (distribution, cumulative, p->numStates);
      //__syncthreads();
      int newState = _drawFromCumulative(cumulative, p->numStates, &rndState);

      states[j] = newState;
    }

    // which configuration is this?

    int config = 0;
    for (int j=0; j < numPotentials; j++){
      config *= potentials[j].numStates;
      config += states[j];
    }
    
    //counts[config] ++;
    atomicAdd(countsBase+config, 1);

  }

}

__global__
void initPotential(Potential*p, 
                    int numStates,
                    float* conditionals,
                    Potential** parents,
                    int numParents) {
  p->numStates = numStates;
  int numConditionals = numStates;

  for (int iParent=0; iParent < numParents; iParent++){
    Potential * parent = parents[iParent];

    numConditionals *= parent->numStates;
    int indexOfChildInParent = parent->numChildren;
    parent->children[indexOfChildInParent] = p;
    parent->indexInChild[indexOfChildInParent] = iParent;
    parent->numChildren++;
  }


  p->numConditionals = numConditionals;
  p->conditionals = conditionals;

  memset (p->parents, 0, sizeof(Potential*) * MAX_PARENTS);
  memset (p->children, 0, sizeof(Potential*) * MAX_CHILDREN);
  memset (p->indexInChild, 0, sizeof(int) * MAX_CHILDREN);

  p->numParents = numParents;
  p->numChildren = 0;
  if (numParents > 0){ //suspicious that we need this or else mem err
    memcpy (p->parents,  parents, numParents * sizeof(Potential*));
  } 
  
  int numDimensions = 1 + numParents;
  p->dimensions[0] = numStates;
  for (int iDim = 1; iDim < numDimensions; iDim++){
    Potential *parent = p->parents[iDim-1];
    
    p->dimensions[iDim] = parent->numStates;
  }
  p->isFrozen = false;
}

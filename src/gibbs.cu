#include <curand_kernel.h>
#include <assert.h>
#include "constants.h"
#include "potential.h"

#define DPRINT(...)

__device__ double rnd(curandState* state) {
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
                 int numDimensions,
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
  for (int i=0; i< n; i++){
    distribution[i] = 1.0f;
  }
}

__device__
void _normalizeDistribution (float* distribution, int n){
  float sum = 0.f;
  for (int i=0; i< n; i++){
    assert(distribution[i] > 0.f);
    sum += distribution[i] ;
  }
  float scale = 1.f / sum;
  for (int i=0; i< n; i++){
    distribution[i] *= scale ;
  }
}

__device__
void _cumulativeDistribution (float* distribution, float * cumulative, int n){
  cumulative[0] = distribution[0];
  for (int i=1; i< n; i++){
    cumulative[i] = cumulative[i-1] + distribution[i] ;
  }
}

__device__
int _drawFromCumulative (float* cumulative, int n, curandState* rndState){
  double r =  rnd(rndState);
  
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
void _conditionalGiven(const Potential *const potentials , int offset, const int * const states, int indexUnfixed, float* distribution){

  const Potential *const potential = potentials + offset;
  int numDimensions = potential->numParents + 1;
  int indices[MAX_DIMENSIONS];

  indices[0] = states[offset];
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
  assert (lengthOut == potential->numStates);
  
  for (int iState =0; iState < potential->numStates; iState++){
    assert (offsetOut + iState * strideOut < potential->numConditionals);
    distribution[iState] *= potential->conditionals[offsetOut + iState * strideOut]; 
  }
}

__global__
void gibbs (const Potential* const potentials, int numPotentials, const int *const initialStates,
            int countsBase[], int numCounts, int numIterations) {
  curandState rndState;
  rndSeed(&rndState);

  if (numPotentials > MAX_POTENTIALS){
    return;
  }
  int states[MAX_POTENTIALS]; 
  memcpy (states, initialStates, numPotentials*sizeof(int));
  
  // int * counts = countsBase + blockIdx.x * numCounts;
  int counts[MAX_CONFIGURATIONS];
  memset ( counts, 0, numCounts* sizeof(int));
    
  for (int i=0; i<numIterations; i++){
    
    for (int j=0; j < numPotentials; j++){
      const Potential *const p = potentials + j;
      if (p->isFrozen){
        continue;
      }
      
      float distribution [MAX_STATES];
      _initDistribution(distribution, p->numStates);

           
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
      
      float cumulative [MAX_STATES];
      _cumulativeDistribution (distribution, cumulative, p->numStates);

      int newState = _drawFromCumulative(cumulative, p->numStates, &rndState);

      states[j] = newState;
    }

    // which configuration is this?

    int config = 0;
    for (int j=0; j < numPotentials; j++){
      config *= potentials[j].numStates;
      config += states[j];
    }
    
    counts[config] ++;
  }

  int * gCounts = countsBase ;//+ blockIdx.x * numCounts;
  for (int i=0; i<numCounts; i++){
    atomicAdd(gCounts+i, counts[i]);
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

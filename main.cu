#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>


#include "gibbs.h"
#include "projection.h"
#include "rnd.h"
#include "potential.h"
#include "cudacall.h"



#ifndef DEBUG
#define DEBUG 0
#endif

#define DPRINT(...)                                             \
  //do { if (DEBUG) fprintf(stderr, __VA_ARGS__); } while (0)
#define ASSERT(...)
#undef assert
#define assert(...)

#define MAX_DIMENSIONS 10
#define MAX_STATES 10

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
  DPRINT("drawFromCumulative %d %f %f\n", n, r, cumulative[n-1]);
  assert (false);
  return 0;
}

__device__
void _conditionalGiven(const Potential *const potentials , int offset, const int * const states, int indexUnfixed, float* distribution){
  ASSERT (numDimensions < MAX_DIMENSIONS);

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
            int counts[], int numCounts, int numIterations) {
  curandState rndState;
  rndSeed(&rndState);

  int* states = (int*) malloc (numPotentials * sizeof(int));
  memcpy (states, initialStates, numPotentials*sizeof(int));

  memset (counts, 0, numCounts* sizeof(int));
    
  for (int i=0; i<numIterations/blockDim.x; i++){
    
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
  free (states);

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

int printDevicePotential (Potential*pd) {
  Potential p;  
  CUDA_CALL(cudaMemcpy ( &p,  pd, sizeof(p), cudaMemcpyDeviceToHost));

  printf("%17s %6d\n",  "numStates", p.numStates);

  if (p.numConditionals >=0 && p.numConditionals < 1000){

    float *conditionals = (float*) malloc(p.numConditionals * sizeof(float));
    CUDA_CALL(cudaMemcpy ( conditionals,  p.conditionals, p.numConditionals * sizeof(p.conditionals[0]), cudaMemcpyDeviceToHost));

    for (int i=0; i< p.numConditionals; i++){
      printf("%11s[%3d] %6.3f\n",  "conditionals", i,  conditionals[i]);
    }
    free(conditionals);
  }
  printf("%17s %6d\n",  "numConditionals", p.numConditionals);
  printf("%17s %6d\n",  "numParents", p.numParents);
  printf("%17s %6d\n",  "numChildren", p.numChildren);
  

  if (p.numParents >=0 && p.numParents <= MAX_PARENTS){
    for (int i=0; i<p.numParents; i++){
      int offset = p.parents[i] - pd;
      printf("%11s[%3d] %6d\n",  "parent", i, offset);
    }
  }
  if (p.numChildren >=0 && p.numChildren <= MAX_CHILDREN){
    for (int i=0; i<p.numChildren; i++){
      int offset = p.children[i] - pd;
      printf("%11s[%3d] %6d\n",  "child", i, offset);
    }
  }
  if (p.numChildren >=0 && p.numChildren <= MAX_CHILDREN){
    for (int i=0; i<p.numChildren; i++){
      printf("%11s[%3d] %6d\n",  "indexInChild", i, p.indexInChild[i]);
    }
  }
  if (p.numParents >=0 && p.numParents <= MAX_PARENTS){
    for (int i=0; i<=p.numParents; i++){
      printf("%11s[%3d] %6d\n",  "dimensions", i, p.dimensions[i]);
    }
  }
  printf("%17s %6d\n",  "isFrozen", p.isFrozen);
  return 0;
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


__device__ double rnd(curandState* state) {
  return curand_uniform( state );
}

__device__ void rndSeed ( curandState * state){
  unsigned int seed = (unsigned int) clock();
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init ( seed, id, 0, &state[id] );
} 


int main (int argc, char ** argv){

  int N = 1;

  const int numPotentials = 5;
  Potential* devPotentials;
  CUDA_CALL(cudaMalloc ( (void**) &devPotentials, numPotentials *sizeof( Potential ) ));
  Potential * da = devPotentials+0, *db = devPotentials+1, *dc=devPotentials+2, *dd=devPotentials+3, *de = devPotentials+4;


  const int numConditionals = 2+4+4+4+8 ;
  float conditionals[numConditionals] = {
    // a
    0.4f, 0.6f,
    //b
    0.3f, 0.7f, 0.8f, 0.2f,
    //c
    0.7f, 0.3f, 0.4f, 0.6f,
    //d
    0.5f, 0.5f, 0.1f, 0.9,
    // e
    0.9f, 0.1f, 0.999f,0.001f,
    0.999f, 0.001f,  0.999f,0.001f};

  float *devConditionals;
  CUDA_CALL(cudaMalloc ( (void**) &devConditionals, numConditionals * sizeof( float ) ));
  CUDA_CALL(cudaMemcpy (devConditionals, conditionals, numConditionals* sizeof(float), cudaMemcpyHostToDevice));
  float *dca = devConditionals, *dcb = dca + 2, *dcc = dcb +4, *dcd = dcc + 4, *dce = dcd + 4;

  printf("devConditionals = %p\n", devConditionals);

  const int numStates[numPotentials] = {2,2,2,2,2};
  Potential * parents[MAX_PARENTS];
  Potential ** devParents;
  CUDA_CALL(cudaMalloc ( (void**) &devParents, MAX_PARENTS * sizeof( Potential * ) ));

   // P(A)
  
  initPotential<<<1, 1>>> (da, numStates[0], dca, 
                  devParents, 0 );
  printDevicePotential(da);

  // P(B|A)
  parents[0] = da;
  CUDA_CALL(cudaMemcpy (devParents, parents,  MAX_PARENTS * sizeof( Potential * ), cudaMemcpyHostToDevice));
  
  initPotential<<<1, 1>>> (db, numStates[1], dcb, 
                 devParents, 1 );
  printDevicePotential(db);



  // P(C|A)
  parents[0] = da;
  CUDA_CALL(cudaMemcpy (devParents, parents,  MAX_PARENTS * sizeof( Potential * ), cudaMemcpyHostToDevice));
  initPotential<<<1, 1>>> (dc, numStates[2], dcc, 
                 devParents, 1 );
  printDevicePotential(dc);
  // P(D|B)
  parents[0] = db;
  CUDA_CALL(cudaMemcpy (devParents, parents,  MAX_PARENTS * sizeof( Potential * ), cudaMemcpyHostToDevice));
  initPotential<<<1, 1>>> (dd, numStates[3], dcd, 
                 devParents, 1 );
  printDevicePotential(dd);
  // P(E|D,C)
  parents[0] = dd;
  parents[1] = dc;
  CUDA_CALL(cudaMemcpy (devParents, parents,  MAX_PARENTS * sizeof( Potential * ), cudaMemcpyHostToDevice));
  initPotential<<<1, 1>>> (de, numStates[4], dce,
                  devParents, 2 );
  printDevicePotential(de);
  
  for (int i=0; i< numPotentials; i++){
    Potential* p = da + i;
    printf ("Potential %c %p:\n", 'A' + i, p);
    printDevicePotential(p);
  }

  //data: B=n, E=n
 
 
  // b->isFrozen = e->isFrozen = true;
  
  int numConfigurations = 1;
  for (int i=0; i< numPotentials; i++){
    numConfigurations *= numStates[i];
  }
  int counts[numConfigurations];
  memset (counts, 0, numConfigurations * sizeof(int));

 // initial config: ynyyn  (we use y=0, n=1)
  int  states [numPotentials] = {0,1,0,0,1};
  int * devStates;
  CUDA_CALL(cudaMalloc((void**)&devStates,  numPotentials* sizeof(int)));
  CUDA_CALL(cudaMemcpy (devStates, states, numPotentials* sizeof(int), cudaMemcpyHostToDevice));

  int * devCounts ;
  CUDA_CALL(cudaMalloc( (void**) &devCounts, numConfigurations* sizeof(int)));
  CUDA_CALL(cudaMemcpy (devCounts, counts, numConfigurations* sizeof(int), cudaMemcpyHostToDevice));

  gibbs<<<1,N>>>(devPotentials, numPotentials, devStates, devCounts, numConfigurations, 100);

  CUDA_CALL(cudaMemcpy ( counts,  devCounts, numConfigurations* sizeof(int), cudaMemcpyDeviceToHost));

  for (int j=0; j < numConfigurations; j++){
    printf("%4d: %4d\n", j, counts[j]);
  }

}

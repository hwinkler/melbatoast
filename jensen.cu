#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <cuda.h>
#include "potential.h"
#include "gibbs.h"
#include "cudacall.h"

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
  float *dca = devConditionals+0, *dcb = dca + 2, *dcc = dcb +4, *dcd = dcc + 4, *dce = dcd + 4;

  const int numStates[numPotentials] = {2,2,2,2,2};
  // P(A)
  initPotential<<<1, 1>>> (da, numStates[0], dca, 
                  (Potential *[]) {NULL}, 0 );
  // P(B|A)
  initPotential<<<1, 1>>> (db, numStates[1], dcb, 
                  (Potential *[]) {da}, 1 );
  // P(C|A)
  initPotential<<<1, 1>>> (dc, numStates[2], dcc, 
                  (Potential *[]) {da}, 1 );
  // P(D|B)
  initPotential<<<1, 1>>> (dd, numStates[3], dcd, 
                  (Potential *[]) {db}, 1 );
  // P(E|D,C)
  initPotential<<<1, 1>>> (de, numStates[4], dce,
                  (Potential *[]) {dd, dc}, 2 );
  
  
  for (int i=0; i< numPotentials; i++){
    printf ("Potential %c %p:\n", 'A' + i, devPotentials+i);
    printDevicePotential(devPotentials+i);
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

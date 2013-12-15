#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>


#include "potential.h"
#include "cudacall.h"
#include "parse.h"


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
#define MAX_POTENTIALS 100
#define MAX_CONFIGURATIONS 1000



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

int freezeDevicePotential(Potential *pd, int frozen){
  Potential p;  
  CUDA_CALL(cudaMemcpy ( &p,  pd, sizeof(p), cudaMemcpyDeviceToHost));
  p.isFrozen = frozen;
  CUDA_CALL(cudaMemcpy ( pd, &p, sizeof(p), cudaMemcpyHostToDevice));

  return 0;
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


const int POTENTIALINFO_BLOCK_SIZE = 1000;

typedef struct PotentialInfo {
  char* name;
  char** parentNames;
  int numStates;
  int  numParents;
  float *table;
  int lengthTable;
  Potential *devPtr;
}PotentialInfo;

struct PotentialInfo **parsedPotentials =
  (PotentialInfo**)malloc(POTENTIALINFO_BLOCK_SIZE * sizeof(PotentialInfo*));
int numParsedPotentials = 0;
  
///
/// Allocate a copy of a string with malloc.
///

char *allocName (char * src){
  int n = strlen (src);
  char * p = (char*) malloc(n);
  strcpy (p, src);
  return p;
}

///
/// The parser calls this function with the info for each parsed potential.
/// We just save the info for later.
///

void parseCallback(char* name, int numStates, char**parents, int numParents, float* table, int lengthTable){
  //printf("callback %s \n numStates %d \n parents[", name, numStates);
  for (int i=0; i< numParents; i++){
    //printf (" %s", parents[i] );
  }
  //printf("]\n table [");
  for (int i=0; i< lengthTable; i++){
    //printf(" %f", table[i]);
  }
  //printf("]\n");

  // save this potential info for later
  PotentialInfo * pi = (PotentialInfo*)calloc(1, sizeof(PotentialInfo));
  if (((numParsedPotentials+1) % POTENTIALINFO_BLOCK_SIZE) == 0){
    parsedPotentials = 
      (PotentialInfo**) realloc(parsedPotentials, 
                               (numParsedPotentials +POTENTIALINFO_BLOCK_SIZE) 
                               * sizeof(PotentialInfo*));
  }
  parsedPotentials[numParsedPotentials++] = pi;
  pi->name = allocName(name);
  pi->parentNames = (char**) calloc(numParents, sizeof(char*));
  for (int i=0; i< numParents; i++){
    pi->parentNames[i] = allocName(parents[i]);
  }
  pi->numStates = numStates;
  pi->numParents = numParents;
  pi->table = (float*)malloc(lengthTable*sizeof(float));
  memcpy (pi->table, table, lengthTable*sizeof(float));
  pi->lengthTable = lengthTable;
  pi->devPtr = NULL;
}

int initOnePotential(Potential * devPotential, PotentialInfo ** ppi, int npi, int ip){
  //printf("initOnePotential\n");
  PotentialInfo* pi = ppi[ip];
  // find its parents
  
  Potential ** devParents;
  CUDA_CALL(cudaMalloc ( (void**) &devParents, MAX_PARENTS * sizeof( Potential * ) ));
  //printf("  initOnePotential -- malloced parents\n");
  int err = 0;
  for ( int iParent=0; iParent < pi->numParents; iParent++){
    char * parentName = pi->parentNames[iParent];
    //printf("  initOnePotential -- parent %s\n", parentName);

    int found = 0;
    for (int i= ip-1; i>=0 ; --i){
      PotentialInfo* candidate = ppi[i];
      if (strncmp (candidate->name, parentName, 64) == 0){
        //printf("  initOnePotential -- found parent %s = %p\n", parentName, candidate->devPtr);
          CUDA_CALL(cudaMemcpy (devParents+iParent, &candidate->devPtr ,  sizeof( Potential * ), cudaMemcpyHostToDevice));
          found = 1;
          //printf("  initOnePotential -- copied parent %s\n", parentName);
          break;
      }
    }
    if(found == 0){
      fprintf(stderr, "for %s, cannot locate parent %s\n", pi->name, parentName);
      err = 1;
      break;
    }
  }
  float *devConditionals = 0;
  if ( err == 0){
    // copy the conditional table over
     //printf("  initOnePotential -- allocing table length %d\n",  pi->lengthTable);
  
    CUDA_CALL(cudaMalloc ( (void**) &devConditionals, pi->lengthTable * sizeof( float ) ));
    CUDA_CALL(cudaMemcpy (devConditionals, pi->table, pi->lengthTable* sizeof(float), 
                          cudaMemcpyHostToDevice));    
  }
  if (err == 0){
    //printf("calling initPotential %p\n numStates=%d\n table=[",devPotential,   pi->numStates);
    for (int i=0; i< pi->lengthTable; i++){
      //printf (" %f", pi->table[i]);
    }
    //printf(" ]\n");

    initPotential<<<1, 1>>> (devPotential, pi->numStates, devConditionals, 
                 devParents, pi->numParents );
  }

  // Remember the devPotential for this parsed potential
  pi->devPtr = devPotential;

  // It is OK to free the devParents as the device code made a copy
  CUDA_CALL(cudaFree ( devParents)); 
  return err;
}

void parseNetwork(const char * fileName){
  FILE * fp1 = fopen(fileName, "r");
  parse(fp1, parseCallback);
  fclose(fp1);
}
  
int* parseStates(const char* fileName){
  int *states =(int*) calloc (numParsedPotentials, sizeof(int));
  FILE * fp2 = fopen(fileName, "r");
  char line[1024];

  int iState = 0;
  while (iState < numParsedPotentials && fgets(line, sizeof(line)-1, fp2)) {
    sscanf (line, "%d", states + iState);
  }
  fclose(fp2);
  return states;
}

int main(int argc, char** argv){
  parseNetwork("jensen.bn");
  int * states = parseStates("jensen.state");
 
  Potential* devPotentials;
  CUDA_CALL(cudaMalloc ( (void**) &devPotentials, numParsedPotentials *sizeof( Potential ) ));

  for (int i=0; i<numParsedPotentials; i++){
    initOnePotential (devPotentials + i, parsedPotentials, numParsedPotentials, i);
    //printDevicePotential(devPotentials + i);
  }

    
  //for (int i=0; i< numParsedPotentials; i++){
    //Potential* p = devPotentials + i;
    //printf ("Potential %c %p:\n", 'A' + i, p);
    //printDevicePotential(p);
  //}

  int numConfigurations = 1;
  for (int i=0; i< numParsedPotentials; i++){
    numConfigurations *= parsedPotentials[i]->numStates;
  }

  // If any state is negative, that is a flag that its state is evidence, i.e. 
  // measured data.
  for (int iState=0; iState < numParsedPotentials; iState++){
    if (states[iState] < 0){
      states[iState] = -states[iState];
      freezeDevicePotential(devPotentials + iState, 1);
    }
  }

 // initial config: ynyyn  (we use y=0, n=1)

  int * devStates;
  CUDA_CALL(cudaMalloc((void**)&devStates,  numParsedPotentials* sizeof(int)));
  CUDA_CALL(cudaMemcpy (devStates, states, numParsedPotentials* sizeof(int), cudaMemcpyHostToDevice));

  free(states);
  states = NULL;

  int * devCounts ;
  CUDA_CALL(cudaMalloc( (void**) &devCounts, numConfigurations* sizeof(int)));
  CUDA_CALL(cudaMemset (devCounts, 0, numConfigurations*   sizeof(int)));

  const int N=1, M=1;
  gibbs<<<N,M>>>(devPotentials, numParsedPotentials, devStates, devCounts, numConfigurations, 100);

  int counts[numConfigurations];
  CUDA_CALL(cudaMemcpy ( counts,  devCounts, numConfigurations*  sizeof(int), cudaMemcpyDeviceToHost));

  for (int j=0; j < numConfigurations; j++){
      printf("%4d: ", j);
    for (int n =0; n< 1; n++){
      printf("%6d", counts[j + n * numConfigurations ]);
    }
    printf("\n");
  }

  CUDA_CALL(cudaFree (devPotentials));
  CUDA_CALL(cudaFree (devStates));
  CUDA_CALL(cudaFree (devCounts));

  
}

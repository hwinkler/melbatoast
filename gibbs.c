#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "potential.h"
#include "projection.h"

#ifndef DEBUG
#define DEBUG 0
#endif

#define DPRINT(...)                                             \
  do { if (DEBUG) fprintf(stderr, __VA_ARGS__); } while (0)


// TODO: including mtwist.h caused duplicate defs in the linker
extern double mt_drand();
extern void mt_seed();

void _initDistribution (float* distribution, int n){
  for (int i=0; i< n; i++){
    distribution[i] = 1.0f;
  }
}
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

void _cumulativeDistribution (float* distribution, float * cumulative, int n){
  cumulative[0] = distribution[0];
  for (int i=1; i< n; i++){
    cumulative[i] = cumulative[i-1] + distribution[i] ;
  }
}

int _drawFromCumulative (float* cumulative, int n){
  double r =  mt_drand();
  
  for (int i=0; i<n; i++){
    
    if ( r < cumulative[i]){
      return i;
    }
  }
  
  assert (false);
  return 0;
}

void _conditionalGiven(Potential *potential ,int indexUnfixed, float* distribution){

  int numDimensions = potential->numParents + 1;
  int indices[numDimensions];

  indices[0] = potential->state;
  for (int iParent=0; iParent < potential->numParents; iParent++){
    assert (iParent+1 < numDimensions);
    indices[iParent+1] = potential->parents[iParent]->state;
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

void _printArray(float * f, int n){
  for (int i=0; i<n; i++){
    
  }
  
}


void gibbs (Potential** potentials, int numPotentials, 
            int counts[], int numCounts, int numIterations) {
  
  mt_seed();

  memset (counts, 0, numCounts* sizeof(int));
  
  for (int i=0; i<numIterations; i++){
    
    

    for (int j=0; j < numPotentials; j++){
      Potential *p = potentials[j];
      if (p->isFrozen){
        continue;
      }
      float distribution [p->numStates];
      _initDistribution(distribution, p->numStates);

      
     
      // Obtain the conditional distribution for the current potential

      Potential* potential = p;

      _conditionalGiven (p, 0, distribution);

      
      
      // Multiply in the distribution for this variable in each child potential
      for (int iChild =0; iChild < p->numChildren; iChild++){
        Potential * child = p->children[iChild];
     
        // add one to indexInChild, since the indexInChild refers to zero based
        // among the parents -- but zer oindex is reserved to the potential's variable
        _conditionalGiven (child, p->indexInChild[iChild] + 1, distribution);
        
        _printArray(distribution, p->numStates);
      }
      
      _normalizeDistribution (distribution, p->numStates);
      
      _printArray(distribution, p->numStates);

      float cumulative [p->numStates];
      _cumulativeDistribution (distribution, cumulative, p->numStates);

      int newState = _drawFromCumulative(cumulative, p->numStates);

      p->state = newState;
    }

    // which configuration is this?

    int config = 0;
    for (int j=0; j < numPotentials; j++){
      config *= potentials[j]->numStates;
      config += potentials[j]->state;
    }
    
    counts[config] ++;
  }

}

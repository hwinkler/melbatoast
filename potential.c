#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>

#include "potential.h"
#include "projection.h"

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

void _drawFromCumulative (float* cumulative, int n){
  
  cumulative[0] = distribution[0];
  for (int i=1; i< n; i++){
    cumulative[i] = cumulative[i-1] + distribution[i] ;
  }
}



void _initPotential(Potential*p, 
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
  memcpy (p->conditionals, conditionals, numConditionals * sizeof(float));
  memset (p->parents, 0, sizeof(Potential*) * MAX_PARENTS);
  memset (p->children, 0, sizeof(Potential*) * MAX_CHILDREN);
  memset (p->indexInChild, 0, sizeof(int) * MAX_CHILDREN);
  memcpy (p->parents,  parents, numParents);
  p->numParents = numParents;
   
  int numDimensions = 1 + numParents;

  p->dimensions[0] = numStates;
  for (int iDim = 1; iDim < numDimensions; iDim++){
    Potential *parent = p->parents[iDim-1];
    p->dimensions[iDim] = parent->numStates;
  }
}
  
void _conditionalGiven(Potential *potential ,int indexUnfixed, float* distribution){

  int numDimensions = potential->numParents + 1;
  int indices[numDimensions];

  indices[0] = potential->state;
  for (int iParent=0; iParent < potential->numParents; iParent++){
    assert (iParent+1 < numDimensions);
    indices[iParent+1] = potential->parents[iParent]->state;
  }
  indices[indexUnfixed] = -1;

  int offsetOut = 0, lengthOut =0, strideOut =0;
  projection ( potential->dimensions, indices, numDimensions,
               &offsetOut, &lengthOut, &strideOut);

  for (int iState =0; iState < potential->numStates; iState++){
    assert (iState * strideOut <  potential->numStates);
    assert (offsetOut + iState * strideOut < potential->numConditionals);
    distribution[iState] *= potential->conditionals[offsetOut + iState * strideOut];
  }
}

int main (int argc, char ** argv) {
  srand(time(NULL));

  float conditionals [2 + 4+ 4 + 4 + 8];
  Potential a,b,c,d,e;

  // P(A)

  _initPotential (&a, 2, (float []){0.4f, 0.6f}, 
                  (Potential *[]) {NULL}, 0 );

  // P(B|A)
  _initPotential (&b, 2, (float []){0.3f, 0.7f, 0.8f, 0.2f}, 
                  (Potential *[]) {&a}, 1 );

  // P(C|A)
  _initPotential (&c, 2, (float []){0.7f, 0.3f, 0.4f, 0.6f}, 
                  (Potential *[]) {&a}, 1 );
 
  // P(D|B)
  _initPotential (&d, 2, (float []){0.5f, 0.5f, 0.1f, 0.9f}, 
                  (Potential *[]) {&b}, 1 );

  // P(E|D,C)
  _initPotential (&e, 2, (float []){
      0.9f,  
        0.1f,  
        0.999f,
        0.001f,
        0.999f,  
        0.001f,  
        0.999f,
        0.001f
        },
    (Potential *[]) {&d, &c}, 1 );


  //data: B=n, E=n
  // initial config: ynyyn  (we use y=0, n=1)

  a.state = 0;
  b.state = 1;
  c.state = 0;
  d.state = 0;
  e.state = 1;

  Potential* potentials[] = {&a, &b, &c, &d, &e}; //causal order
  const int numPotentials = 5;
  

  const int numConfigurations = 100;
  for (int i=0; i<numConfigurations; i++){
    for (int j=0; j < numPotentials; j++){
      Potential *p = potentials[j];
      float distribution [p->numStates];
      _initDistribution(distribution, p->numStates);

      // Obtain the conditional distribution for the current potential

      Potential* potential = p;
      _conditionalGiven (p, 0, distribution);
      
      // Multiply in the distribution for this variable in each child potential
      for (int iChild =0; iChild < p->numChildren; iChild++){
        Potential * child = p->children[iChild];
        _conditionalGiven (child, p->indexInChild[iChild], distribution);
      }
      
      _normalizeDistribution (distribution, p->numStates);
      float cumulative [p->numStates];
      _cumulativeDistribution (distribution, cumulative, p->numStates);

      int newState = _drawFromCumulative(cumulative, p->numStates);

    }
    
  }

}

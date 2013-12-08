#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "potential.h"
#include "projection.h"

#define DPRINTF printf

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
  printf ("rand = %f, n = %d\n", r, n);
  for (int i=0; i<n; i++){
    printf("cumulative[%d] = %f\n", i, cumulative[i]);
    if ( r < cumulative[i]){
      return i;
    }
  }
  
  assert (false);
  return 0;
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
  p->conditionals = conditionals;
  memset (p->parents, 0, sizeof(Potential*) * MAX_PARENTS);
  memset (p->children, 0, sizeof(Potential*) * MAX_CHILDREN);
  memset (p->indexInChild, 0, sizeof(int) * MAX_CHILDREN);

  memcpy (p->parents,  parents, numParents * sizeof(Potential*));
  p->numParents = numParents;
   
  printf("initialized all but dimensions\n");
  int numDimensions = 1 + numParents;
  p->dimensions[0] = numStates;
  for (int iDim = 1; iDim < numDimensions; iDim++){
    Potential *parent = p->parents[iDim-1];
    printf("setting dim %d  = %d\n", iDim, parent->numStates);
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
  printf("starting\n");

  mt_seed();

  float conditionals[2+4+4+4+8];
  Potential a,b,c,d,e;

  printf("Initializing potentials\n");
  // P(A)

  float *ca = conditionals+ 0;
  memcpy (ca,( (float []){0.4f, 0.6f}), 2 * sizeof(float));
  float *cb = ca + 2;
  memcpy (cb,( (float []){0.3f, 0.7f, 0.8f, 0.2f}), 4 * sizeof(float));
  float *cc = cb + 4;
  memcpy (cc,( (float []){0.7f, 0.3f, 0.4f, 0.6f}), 4 * sizeof(float));
  float *cd = cc + 4;
  memcpy (cd,( (float []){0.5f, 0.5f, 0.1f, 0.9}), 4 * sizeof(float));
  float *ce = cd + 4;
  memcpy (ce,( (float []) {
        0.9f, 0.1f, 0.999f,0.001f,
          0.999f, 0.001f,  0.999f,0.001f}), 8 * sizeof(float));

  _initPotential (&a, 2, ca, 
                  (Potential *[]) {NULL}, 0 );

  printf("P(A)\n");
 
  // P(B|A)
  _initPotential (&b, 2, cb, 
                  (Potential *[]) {&a}, 1 );
  printf("P(B|A)\n");

  // P(C|A)
  _initPotential (&c, 2, cc, 
                  (Potential *[]) {&a}, 1 );
   printf("P(C|A)\n");

  // P(D|B)
  _initPotential (&d, 2, cd, 
                  (Potential *[]) {&b}, 1 );
  printf("P(D|B)\n");

  // P(E|D,C)
  _initPotential (&e, 2, ce,
    (Potential *[]) {&d, &c}, 1 );
  printf("P(E|D,C)\n");

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

      printf("i=%d, j=%d\n", i, j);

      // Obtain the conditional distribution for the current potential

      Potential* potential = p;
      _conditionalGiven (p, 0, distribution);

      printf("got initial distribution\n");
      
      // Multiply in the distribution for this variable in each child potential
      for (int iChild =0; iChild < p->numChildren; iChild++){
        Potential * child = p->children[iChild];
        _conditionalGiven (child, p->indexInChild[iChild], distribution);
        printf("got  distribution for child %d\n", iChild);
      }
      
      _normalizeDistribution (distribution, p->numStates);
      printf("normalized  distribution \n");

      float cumulative [p->numStates];
      _cumulativeDistribution (distribution, cumulative, p->numStates);
      printf("cululative  distribution \n");

      int newState = _drawFromCumulative(cumulative, p->numStates);

      printf("new state %d\n", newState);

      p->state = newState;
    }
    
  }

}

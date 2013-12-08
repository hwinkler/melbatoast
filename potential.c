#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "potential.h"
#include "projection.h"

void _initDistribution (float* distribution, int n){
  for (int i=0; i< n; i++){
    distribution[i] = 1.0f;
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
  

int main (int argc, char ** argv) {
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
  State sa, sb, sc, sd, se;
  sa.potential = &a; sa.state = 0;
  sb.potential = &b; sb.state = 1;
  sc.potential = &c; sc.state = 0;
  sd.potential = &d; sd.state = 0;
  se.potential = &e; se.state = 1;

  Potential* potentials[] = {&a, &b, &c, &d, &e}; //causal order
  const int numPotentials = 5;
  

  State* states[] = {&sa, &sb, &sc, &sd, &se}; //causal order
 
  const int numConfigurations = 100;
  for (int i=0; i<numConfigurations; i++){
    for (int j=0; j < numPotentials; j++){
      Potential *p = potentials[j];
      float distribution [p->numStates];
      _initDistribution(distribution, p->numStates);

      // Obtain the conditional distribution for the current potential 
      int indices[p->numDimensions];
      indices[0] = -1;
      for (int i=0; i< p->numParents; i++){
        indices[i+1] = p->state; 
      }


      int offsetOut = 0, lengthOut, strideOut;
      projection ( p->dimensions, indices, p->numDimensions,
                   &offsetOut, &lengthOut, &strideOut);

     
    }
    
  }

}

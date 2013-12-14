#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include "potential.h"

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

  memcpy (p->parents,  parents, numParents * sizeof(Potential*));
  p->numParents = numParents;
   
  
  int numDimensions = 1 + numParents;
  p->dimensions[0] = numStates;
  for (int iDim = 1; iDim < numDimensions; iDim++){
    Potential *parent = p->parents[iDim-1];
    
    p->dimensions[iDim] = parent->numStates;
  }
  p->isFrozen = false;
}
  

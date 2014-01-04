#pragma once
#include "constants.h"

class Potential {
public:
  int   numStates;
  const float * __restrict conditionals;
  int numConditionals;   // this is here mainly for debugging
  int numParents;
  int numChildren;
  struct Potential* parents[MAX_PARENTS];
  struct Potential* children[MAX_CHILDREN];
  int indexInChild [MAX_CHILDREN];
  int dimensions [MAX_PARENTS + 1];
  int isFrozen; 
};

//int printDevicePotential (Potential*p);


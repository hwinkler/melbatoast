#pragma once
#include "constants.h"

typedef struct __align__(8) Potential {
  int   numStates;
  float *conditionals;
  int numConditionals;   // this is here mainly for debugging
  int numParents;
  int numChildren;
  struct Potential* parents[MAX_PARENTS];
  struct Potential* children[MAX_CHILDREN];
  int indexInChild [MAX_CHILDREN];
  int dimensions [MAX_PARENTS + 1];
  int isFrozen; 
} Potential;

//int printDevicePotential (Potential*p);


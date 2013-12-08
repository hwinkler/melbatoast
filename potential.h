#pragma once

#define MAX_PARENTS 5
#define MAX_CHILDREN 5

typedef struct Potential {
  int   numStates;
  float *conditionals;
  int numConditionals;   // this is here mainly for debugging
  int numParents;
  int numChildren;
  struct Potential* parents[MAX_PARENTS];
  struct Potential* children[MAX_CHILDREN];
  int indexInChild [MAX_CHILDREN];
  int dimensions [MAX_PARENTS + 1];

  int state;  /// TODO remove. This is the only mutable field.
  int isFrozen;

} Potential;


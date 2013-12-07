#pragma once

#define MAX_PARENTS 5
#define MAX_CHILDREN 5


typedef struct Potential {
  int   numStates;
  float *conditionals;
  int numParents;
  int numChildren;
  struct Potential* parents[MAX_PARENTS ];
  struct Potential* children[MAX_CHILDREN];
} Potential;

typedef struct State {
  Potential* potential;
  int state;
} State;

typedef struct PotentialRef {
  Potential* potential;
  int dimensionIndex;
} PotentialRef;

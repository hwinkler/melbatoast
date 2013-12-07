#pragma once

#define MAX_PARENTS 5
#define MAX_CHILDREN 5
#define MAX_STATES 10

typedef struct Potential {
  int   numStates;
  float *conditionals;
  // struct Potential* parents[MAX_PARENTS +1];
} Potential;

typedef struct State {
  Potential* potential;
  int state;
} State;

typedef struct PotentialRef {
  Potential* potential;
  int dimensionIndex;
} PotentialRef;

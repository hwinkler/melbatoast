#pragma once

#define MAX_PARENTS 5

typedef struct Potential {
  int   numStates;
  float *conditionals;
  struct Potential* parents[MAX_PARENTS];
} Potential;

typedef struct State {
  Potential* potential;
  int state;
} State;


int sample (Potential* potential, State* states[], int numStates);

void projection (
                 int *dimensions,
                 int *indices,
                 int numDimensions,
                 int *offsetOut,
                 int *lengthOut,
                 int *strideOut);

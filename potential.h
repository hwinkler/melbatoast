#pragma once
const int MAX_PARENTS=5;

typedef struct Potential {
  int   numStates;
  float *conditionals;
  Potential* parents[MAX_PARENTS];
} Potential;

typedef struct State {
  Potential* potential;
  int state;
} State;


int sample (Potential* potential, State* states[], int numStates);


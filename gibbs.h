#pragma once
#include "potential.h"
__global__
extern void gibbs (const Potential* const potentials, int numPotentials, const int * const initialStates,
                   int counts[], int numCount, int numIterations);


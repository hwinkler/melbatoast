#pragma once
#include "potential.h"
__device__
extern void gibbs (const Potential* const potentials, int numPotentials, const int * const initialStates,
                   int counts[], int numCount, int numIterations,  int seeds[]);


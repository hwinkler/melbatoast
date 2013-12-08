#pragma once
#include "potential.h"
extern void gibbs (Potential** potentials, int numPotentials, 
                   int counts[], int numCount, int numIterations);


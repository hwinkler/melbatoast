#pragma once

__global__
void initPotential(Potential*p, 
                    int numStates,
                    float* conditionals,
                    Potential** parents,
                   int numParents);

__global__
void gibbs (const Potential* const potentials, int numPotentials, const int *const initialStates,
            int countsBase[], int numCounts, int numIterations);

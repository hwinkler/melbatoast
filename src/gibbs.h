#pragma once

__global__
void initPotential(Potential*p, 
                    int numStates,
                    float* conditionals,
                    Potential** parents,
                   int numParents);

__global__
void gibbs ( const Potential* __restrict__ potentials, int numPotentials, const int *__restrict__  initialStates,
            int countsBase[], int numCounts, int numIterations);

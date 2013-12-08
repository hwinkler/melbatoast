#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "../potential.h"
#include "../gibbs.h"

int main (int argc, char ** argv){

  const int numConditionals = 2+4+4+4+8 ;
  float conditionals[numConditionals];
  Potential potentials [5]; //causal order
  Potential * a = potentials+0, *b = potentials+1, *c=potentials+2, *d=potentials+3, *e = potentials+4;


  float *ca = conditionals+ 0;
  memcpy (ca,( (float []){0.4f, 0.6f}), 2 * sizeof(float));
  float *cb = ca + 2;
  memcpy (cb,( (float []){0.3f, 0.7f, 0.8f, 0.2f}), 4 * sizeof(float));
  float *cc = cb + 4;
  memcpy (cc,( (float []){0.7f, 0.3f, 0.4f, 0.6f}), 4 * sizeof(float));
  float *cd = cc + 4;
  memcpy (cd,( (float []){0.5f, 0.5f, 0.1f, 0.9}), 4 * sizeof(float));
  float *ce = cd + 4;
  memcpy (ce,( (float []) {
        0.9f, 0.1f, 0.999f,0.001f,
          0.999f, 0.001f,  0.999f,0.001f}), 8 * sizeof(float));

  // P(A)

  initPotential (a, 2, ca, 
                  (Potential *[]) {NULL}, 0 );

  
 
  // P(B|A)
  initPotential (b, 2, cb, 
                  (Potential *[]) {a}, 1 );
  

  // P(C|A)
  initPotential (c, 2, cc, 
                  (Potential *[]) {a}, 1 );
   

  // P(D|B)
  initPotential (d, 2, cd, 
                  (Potential *[]) {b}, 1 );
  

  // P(E|D,C)
  initPotential (e, 2, ce,
                  (Potential *[]) {d, c}, 2 );
  

  //data: B=n, E=n
  // initial config: ynyyn  (we use y=0, n=1)

  int  states [5] = {0,1,0,0,1};

  // b->isFrozen = e->isFrozen = true;

  const int numPotentials = 5;


  int numPossibleConfigurations = 1;
  for (int i=0; i< numPotentials; i++){
    numPossibleConfigurations *= potentials[i].numStates;
  }
  int counts[numPossibleConfigurations];

  gibbs(potentials, numPotentials, states, counts, numPossibleConfigurations, 1000000);


  for (int j=0; j < numPossibleConfigurations; j++){
    printf("%4d: %4d\n", j, counts[j]);
  }

}

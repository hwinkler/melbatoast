#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "potential.h"
#include "projection.h"

void _initDistribution (float* distribution, int n){
  for (int i=0; i< n; i++){
    distribution[i] = 1.0f;
  }
}

void _initPotential(Potential*p, 
                   int numStates,
                   float* conditionals,
                   Potential** parents,
                   int numParents) {
  p->numStates = numStates;
  int numConditionals = numStates;
  for (int iParent=0; iParent < numParents; iParent++){
    numConditionals *= parents[iParent]->numStates;
  }
  memcpy (p->conditionals, conditionals, numConditionals * sizeof(float));
}
  

int main (int argc, char ** argv) {
  float conditionals [2 + 4+ 4 + 4 + 8];
  Potential a,b,c,d,e;

  // P(A)

  _initPotential (&a, 2, (float []){0.4f, 0.6f}, 
                  (Potential *[]) {NULL}, 0 );

  // P(B|A)
  _initPotential (&b, 2, (float []){0.3f, 0.7f, 0.8f, 0.2f}, 
                  (Potential *[]) {&a}, 1 );

  // P(C|A)
  _initPotential (&c, 2, (float []){0.7f, 0.3f, 0.4f, 0.6f}, 
                  (Potential *[]) {&a}, 1 );

 
  // P(D|B)
  _initPotential (&d, 2, (float []){0.5f, 0.5f, 0.1f, 0.9f}, 
                  (Potential *[]) {&b}, 1 );

  // P(E|D,C)
  _initPotential (&e, 2, (float []){
      0.9f,  
        0.1f,  
        0.999f,
        0.001f,
        0.999f,  
        0.001f,  
        0.999f,
        0.001f
        },
    (Potential *[]) {&d, &c}, 1 );


  //data: B=n, E=n
  // initial config: ynyyn  (we use y=0, n=1)
  State sa, sb, sc, sd, se;
  sa.potential = &a; sa.state = 0;
  sb.potential = &b; sb.state = 1;
  sc.potential = &c; sc.state = 0;
  sd.potential = &d; sd.state = 0;
  se.potential = &e; se.state = 1;

  const int numConfigurations = 100;
  Potential* potentials[] = {&a, &b, &c, &d, &e}; //causal order
  const int numPotentials = 5;
  
  // Each random variable is the first, or most rapidly-varying, 
  // index within its own potential. We need a lookup table for
  // each RV telling its index in the potentials in which it participates
  // as a child or a parent.

  for (int i=0; i<numPotentials; i++){
    Potential *p = potentials[i];
    PotentialRef refs [MAX_CHILDREN];
    
    
  }

  State* states[] = {&sa, &sb, &sc, &sd, &se}; //causal order
 
  for (int i=0; i<numConfigurations; i++){
    
    for (int j=0; j < numPotentials; j++){
      Potential *p = potentials[j];
      float distribution [MAX_STATES];
      _initDistribution(distribution, MAX_STATES);
      
      PotentialRef** potentialsInvolving ;
      int numPotentialsInvolving;
      for (int ip =0; ip < numPotentialsInvolving; ip++){
        int offset=0, length=0, stride = 0;
        PotentialRef* ref = potentialsInvolving[ip];
        int myIndex = ref->dimensionIndex;
        Potential * otherPotential = ref->potential;
        
      }
     
    }
    
  }

}

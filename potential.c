#include <stdbool.h>
#include <stddef.h>
#include "potential.h"

bool _isParent(Potential* child, Potential* test){
  Potential* parent = NULL;
  for (int i = 0; 
       parent != NULL && i < MAX_PARENTS;
       i++){
    parent = child->parents[i];
    if (parent == test){
      return true;
    }
  }
  return false;
}

int sample (Potential* potential, State* states[],int numStates) {
  for (int i=0; i< numStates; i++){
    Potential* statePotential =  states[i]->potential;
    if (_isParent (potential, statePotential)){
      //statePotential is one of our parents
      
    } else if (_isParent (statePotential, potential)) {
      //statePotential is one of our children
      
        
    }
  }
  return -1;
}

#include <stddef.h>
#include <string.h>
#include "projection.h"

int main (int argc, char ** argv) {
  float conditionals [2 + 4+ 4 + 4 + 8];
  Potential a,b,c,d,e;

  // P(A)
  a.numStates = 2;
  a.conditionals = conditionals + 0;
  a.conditionals[0] = 0.4f;
  a.conditionals[1] = 0.6f;
  memset (a.parents, 0, sizeof(Potential*) * (MAX_PARENTS+1)) ;
  
  // P(B|A)
  b.numStates = 2;
  b.conditionals = a.conditionals + a.numStates;
  b.conditionals[0] = 0.3f;     b.conditionals[2] = 0.8f;
  b.conditionals[1] = 0.7f;     b.conditionals[3] = 0.2f;
  memset (b.parents, 0, sizeof(Potential*) * (MAX_PARENTS+1)) ;
  b.parents[0] = &a;

  // P(C|A)
  c.numStates = 2;
  c.conditionals = b.conditionals + a.numStates * b.numStates;
  c.conditionals[0] = 0.3f;     c.conditionals[2] = 0.8f;
  c.conditionals[1] = 0.7f;     c.conditionals[3] = 0.2f;
  memset (c.parents, 0, sizeof(Potential*) * (MAX_PARENTS+1)) ;
  c.parents[0] = &a;

  // P(D|B)
  d.numStates = 2;
  d.conditionals = c.conditionals + a.numStates * c.numStates;
  d.conditionals[0] = 0.5f;     d.conditionals[2] = 0.1f;
  d.conditionals[1] = 0.5f;     d.conditionals[3] = 0.9f;
  memset (d.parents, 0, sizeof(Potential*) * (MAX_PARENTS+1)) ;
  d.parents[0] = &b;

  // P(E|D,C)
  e.numStates = 2;
  e.conditionals = d.conditionals + b.numStates * d.numStates;
  e.conditionals[0] = 0.9f;    
  e.conditionals[1] = 0.1f;    
  e.conditionals[2] = 0.999f;
  e.conditionals[3] = 0.001f;
  e.conditionals[4] = 0.999f;    
  e.conditionals[5] = 0.001f;    
  e.conditionals[6] = 0.999f;
  e.conditionals[7] = 0.001f;

  memset (e.parents, 0, sizeof(Potential*) * (MAX_PARENTS+1)) ;
  e.parents[0] = &d;
  e.parents[1] = &c;

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
  for (int i=0; i<numConfigurations; i++){
    
    for (int j=0; j < numPotentials; j++){
      Potential *p = potentials[j];
      float distribution [MAX_STATES];
      for (int k=0; k< MAX_STATES; k++){
        distribution[i] = 1.0f;
      }

      // for every parent
      for (int iParent =0; iParent<MAX_PARENTS && p->parents[iParent] != NULL; iParent++){
        Potential *parent = p->parents[iParent];
        

      }
      
      int offset=0, length=0, stride = 0;
     
    }
    
  }

}

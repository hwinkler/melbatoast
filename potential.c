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

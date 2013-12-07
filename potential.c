#include <stdbool.h>
#include "potential.h"

bool _isParent(Potential* child, Potential* test){
  for (Potential* parent = child.parents[0]; 
       parent!= null && (parent - child.parents < MAX_PARENTS);
       parent ++){
    if (parent == test){
      return true;
    }
  }
  return false;
}

int sample (Potential* potential, State* states[],int numStates) {
  for (int i=0; i<numStates, i++){
    Potential* statePotential =  states[i].potential;
    if (_isParent (potential, statePotential)){
      //statePotential is one of our parents

    } else if (_isParent (statePotential, potential) {
      //statePotential is one of our children


    }
  }

}

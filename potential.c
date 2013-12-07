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

void projection (
                 int *dimensions,
                 int *indices,
                 int numDimensions,
                 int *offsetOut,
                 int *lengthOut,
                 int *strideOut) {

  // the first dimension varies fastest
  // index -1 indicates the dimension we are extracting

  int hypersheetSize = 1;
  int stride = -1;
  int index0 = 0;
  int length = -1;
  
  for (int i =0; i < numDimensions ; i++){

    int index;
    if (indices[i] < 0){
      index = 0;
      stride = hypersheetSize;
      length = dimensions[i];
    } else {
      index = indices[i];
    }
    index0 += hypersheetSize * index;
    hypersheetSize *= dimensions[i];
  }

  *offsetOut = index0;
  *lengthOut = length;
  *strideOut = stride;

}



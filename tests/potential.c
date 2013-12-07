#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include "../potential.h"

int _verify (int* dimensions,int* indices, int* expected){
  int offsetOut=-1, lengthOut=-1, strideOut=-1;
  projection(dimensions, indices, 3, &offsetOut, &lengthOut, &strideOut);
  if (offsetOut != expected[0] || lengthOut != expected[1] || strideOut != expected[2]){
    printf ("error:\n");
    printf ("dimensions %2d %2d %2d\n", dimensions[0], dimensions[1], dimensions[2]);
    printf ("indices    %2d %2d %2d\n", indices[0], indices[1], indices[2]);
    printf ("expected   %2d %2d %2d\n", expected[0], expected[1], expected[2]);
    printf ("actual     %2d %2d %2d\n", offsetOut, lengthOut, strideOut);
    return 0;
  } else {
    return 1;
  }
}
   
  

int main (){
  printf("running test potential\n");
  const int na = 5, nb = 5, nc = 4;
  
  int dimensions[3] = {na, nb, nc};
  int result = true;

  int i1[3] = {-1, 0, 0};
  int e1[3] = {0, na, 1};
  result = result && _verify (dimensions, i1, e1);

  int i2[3] = {0, -1, 0};
  int e2[3] = {0, na, nb};
  result = result && _verify (dimensions, i2, e2);

  if (result){
    printf("passed\n");
  } else {
    printf("failed\n");
  }
}

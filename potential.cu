#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "potential.h"



__global__ void add(int *a, int *b, int *c) { 
  *c = *a + *b;
}


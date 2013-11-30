#include <stdio.h>

__global__ void add (int *a, int *b, int *c) {
  *c = *a + *b;
}

void check (const char* msg, int e){
  printf ("%s: %d\n", msg, e);
}

int main (void){
  int a, b, c;
  int *d_a, *d_b, *d_c;
  int size = sizeof(int);

  check("cudaMalloc a", cudaMalloc((void**)&d_a, size));
  check("cudaMalloc b",cudaMalloc((void**)&d_b, size));
  check("cudaMalloc c",cudaMalloc((void**)&d_c, size));
  
  a = 2;
  b = 7;
  
  check("cudaMemcpy d_a<-a", cudaMemcpy (d_a, &a, size, cudaMemcpyHostToDevice));
  check("cudaMemcpy d_b<-b", cudaMemcpy (d_b, &b, size, cudaMemcpyHostToDevice));

  add<<<1,1>>>(d_a, d_b, d_c);

  check("cudaMemcpy c<-d_c", cudaMemcpy (&c, d_c, size, cudaMemcpyDeviceToHost));
  check("cudaFree a", cudaFree(d_a));
  check("cudaFree b", cudaFree(d_b));
  check("cudaFree c", cudaFree(d_c));
  printf ("c= %d\n", c);
  return 0;
}


#include <stdio.h>

__global__ void add (int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  c[idx] = a[idx] + b[idx];
}

void check (const char* msg, int e){
  printf ("%s: %d\n", msg, e);
}

int main (void){

  const int N = 512; // number of vector elements  
  const int M = 8; // threads per block

  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N*sizeof(int);

  a = (int*) calloc(N, sizeof(int));
  b = (int*) calloc(N, sizeof(int));
  c = (int*) calloc(N, sizeof(int));
  
  check("cudaMalloc d_a", cudaMalloc((void**)&d_a, size));
  check("cudaMalloc d_b",cudaMalloc((void**)&d_b, size));
  check("cudaMalloc d_c",cudaMalloc((void**)&d_c, size));
  
  for (int i =0; i<N; i++){
    a[i] = i;
    b[i] = N/2 - i;
  }
  
  check("cudaMemcpy d_a<-a", cudaMemcpy (d_a, a, size, cudaMemcpyHostToDevice));
  check("cudaMemcpy d_b<-b", cudaMemcpy (d_b, b, size, cudaMemcpyHostToDevice));

  add<<<N,M>>>(d_a, d_b, d_c);

  check("cudaMemcpy c<-d_c", cudaMemcpy (c, d_c, size, cudaMemcpyDeviceToHost));
  check("cudaFree a", cudaFree(d_a));
  check("cudaFree b", cudaFree(d_b));
  check("cudaFree c", cudaFree(d_c));
  for (int i =0; i<N; i++){
    if (c[i] != a[i] + b[i] ){
      printf ("c[%d]= %d, should be %d\n", i, c[i], a[i] + b[i]);
      return 1;
    }
  }
  printf("ok\n");
  return 0;
}


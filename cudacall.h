#pragma once

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error %d at %s:%d;  %s\n",x, __FILE__,__LINE__,   cudaGetErrorString(x)); \
      return EXIT_FAILURE;}} while(0)

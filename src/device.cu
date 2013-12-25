#include <stdio.h>
#include "device.h"
#include "cudacall.h"

void printDeviceProp (cudaDeviceProp* prop){
  const char * FORMATD="%30s %9d\n";
  const char * FORMATD2="%30s %9d %9d\n";
  const char * FORMATD3="%30s %9d %9d %9d\n";
  const char * FORMATU="%30s %9u\n";
 
  printf(FORMATD,"asyncEngineCount",prop->asyncEngineCount);
  printf(FORMATD,"canMapHostMemory",prop->canMapHostMemory);
  printf(FORMATD,"clockRate",prop->clockRate);
  printf(FORMATD,"computeMode",prop->computeMode);
  printf(FORMATD,"concurrentKernels",prop->concurrentKernels);
  printf(FORMATD,"deviceOverlap",prop->deviceOverlap);
  printf(FORMATD,"ECCEnabled",prop->ECCEnabled);
  printf(FORMATD,"integrated",prop->integrated);
  printf(FORMATD,"kernelExecTimeoutEnabled",prop->kernelExecTimeoutEnabled);
  printf(FORMATD,"l2CacheSize",prop->l2CacheSize);
  printf(FORMATD,"major",prop->major);
  printf(FORMATD3,"maxGridSize",prop->maxGridSize[0],prop->maxGridSize[1],prop->maxGridSize[2]); 
  printf(FORMATD,"maxSurface1D",prop->maxSurface1D);
  printf(FORMATD2,"maxSurface1DLayered",prop->maxSurface1DLayered[0],prop->maxSurface1DLayered[1]); 
  printf(FORMATD2,"maxSurface2D",prop->maxSurface2D[0],prop->maxSurface2D[1]); 
  printf(FORMATD3,"maxSurface2DLayered",prop->maxSurface2DLayered[0],prop->maxSurface2DLayered[1],prop->maxSurface2DLayered[2]); 
  printf(FORMATD3,"maxSurface3D",prop->maxSurface3D[0],prop->maxSurface3D[1],prop->maxSurface3D[2]);
  printf(FORMATD,"maxSurfaceCubemap",prop->maxSurfaceCubemap);
  printf(FORMATD2,"maxSurfaceCubemapLayered",prop->maxSurfaceCubemapLayered[0],prop->maxSurfaceCubemapLayered[1]); 
  printf(FORMATD,"maxTexture1D",prop->maxTexture1D);
  printf(FORMATD2,"maxTexture1DLayered",prop->maxTexture1DLayered[0],prop->maxTexture1DLayered[1]); 
  printf(FORMATD,"maxTexture1DLinear",prop->maxTexture1DLinear);
  printf(FORMATD2,"maxTexture2D",prop->maxTexture2D[0],prop->maxTexture2D[1]); 
  printf(FORMATD2,"maxTexture2DGather",prop->maxTexture2DGather[0],prop->maxTexture2DGather[1]); 
  printf(FORMATD2,"maxTexture2DLayered",prop->maxTexture2DLayered[0],prop->maxTexture2DLayered[1]); 
  printf(FORMATD3,"maxTexture2DLinear",prop->maxTexture2DLinear[0],prop->maxTexture2DLinear[1],prop->maxTexture2DLinear[2]); 
  printf(FORMATD3,"maxTexture3D",prop->maxTexture3D[0],prop->maxTexture3D[1],prop->maxTexture3D[2]); 
  printf(FORMATD,"maxTextureCubemap",prop->maxTextureCubemap);
  printf(FORMATD2,"maxTextureCubemapLayered",prop->maxTextureCubemapLayered[0],prop->maxTextureCubemapLayered[1]); 
  printf(FORMATD3,"maxThreadsDim",prop->maxThreadsDim[0],prop->maxThreadsDim[1],prop->maxThreadsDim[2]); 
  printf(FORMATD,"maxThreadsPerBlock",prop->maxThreadsPerBlock);
  printf(FORMATD,"maxThreadsPerMultiProcessor",prop->maxThreadsPerMultiProcessor);
  printf(FORMATD,"memoryBusWidth",prop->memoryBusWidth);
  printf(FORMATD,"memoryClockRate",prop->memoryClockRate);
  printf(FORMATU,"memPitch",prop->memPitch);
  printf(FORMATD,"minor",prop->minor);
  printf(FORMATD,"multiProcessorCount",prop->multiProcessorCount);
  printf(FORMATD,"name", prop->name);
  printf(FORMATD,"pciBusID",prop->pciBusID);
  printf(FORMATD,"pciDeviceID",prop->pciDeviceID);
  printf(FORMATD,"pciDomainID",prop->pciDomainID);
  printf(FORMATD,"regsPerBlock",prop->regsPerBlock);
  printf(FORMATU,"sharedMemPerBlock",prop->sharedMemPerBlock);
  printf(FORMATU,"surfaceAlignment",prop->surfaceAlignment);
  printf(FORMATD,"tccDriver",prop->tccDriver);
  printf(FORMATU,"textureAlignment",prop->textureAlignment);
  printf(FORMATU,"texturePitchAlignment",prop->texturePitchAlignment);
  printf(FORMATU,"totalConstMem",prop->totalConstMem);
  printf(FORMATU,"totalGlobalMem",prop->totalGlobalMem);
  printf(FORMATD,"unifiedAddressing",prop->unifiedAddressing);
  printf(FORMATD,"warpSize",prop->warpSize);
 
}
int selectGPU(int verboseFlag){
  int num_devices, device;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices > 1) {
    int max_multiprocessors = 0, max_device = 0;
    for (device = 0; device < num_devices; device++) {
      cudaDeviceProp properties;
      CUDA_CALL(cudaGetDeviceProperties(&properties, device));

      if (verboseFlag){
        printf ("device %d:", device);
        printDeviceProp(&properties);
      }
      if (max_multiprocessors < properties.multiProcessorCount) {
        max_multiprocessors = properties.multiProcessorCount;
        max_device = device;
      }
    }
    CUDA_CALL(cudaSetDevice(max_device));
    if (verboseFlag){
      printf ("using device %d having %d multiprocessors", max_device, max_multiprocessors);
    }
  }

  return 0;
}

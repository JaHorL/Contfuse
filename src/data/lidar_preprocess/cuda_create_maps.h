#ifndef _CUDACREATEMAPS_H
#define _CUDACREATEMAPS_H

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

#include "data_types.h"



__device__ static float atomicMax(float* address, float val);

__global__ void CreatePreBevMapOnGPU(float *bev_d, const float *pts_d,
                                     const int pts_num,  PreprocessParams params);

__global__ void CreatePreFusionIdxMapOnGPU(float *mapping1x_d, float *mapping2x_d, 
                                           float *mapping4x_d, float *mapping8x_d, 
                                           const float *pts_d, const int pts_num,
                                           PreprocessParams params);

__global__ void CreateBevMapOnGPU(float *bev_flip_d, PreprocessParams params);

__global__ void CreateFusionIdxMapOnGPU(float *pre_mapping, float *mapping, const float *tr, 
                                        const int downsample_ratio,  PreprocessParams params);

#endif
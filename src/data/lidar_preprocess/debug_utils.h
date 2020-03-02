#ifndef _DEBUGUTILS_H
#define _DEBUGUTILS_H
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "data_types.h"


void PrintArray(const float* array, int size);

void ParamsPrint(const PreprocessParams &params, const MemorySize &mz);

#endif //_DEBUGUTILS_H
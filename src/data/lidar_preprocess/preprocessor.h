#ifndef _PREPROCESSOR_H
#define _PREPROCESSOR_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include "boost/python.hpp"
#include "boost/python/numpy.hpp"

#include "debug_utils.h"
#include "cuda_create_maps.h"
#include "data_types.h"
#include "timer.h"


namespace bp = boost::python;
namespace bn = boost::python::numpy;


double cpuSecond();

class Preprocessor
{
public:
  Preprocessor() {
  }
  ~Preprocessor() {
    FreeMemory();
  }
  // Preprocessor(const Preprocessor&) = delete;
  // Preprocessor &operator=(const Preprocessor&) = delete;
  void PreprocessorInit(float bev_x_min,        float bev_x_max,
                        float bev_y_min,        float bev_y_max,
                        float bev_z_min,        float bev_z_max,
                        float bev_x_resolution, float bev_y_resolution,
                        float bev_z_resolution, size_t sat_z,
                        size_t h,               size_t w,
                        float h_scale,          float w_scale);
  void PreprocessData(const bn::ndarray &lidar, const bn::ndarray &tr, size_t pts_num);
  bn::ndarray GetBev();
  bn::ndarray GetMapping1x();
  bn::ndarray GetMapping2x();
  bn::ndarray GetMapping4x();
  bn::ndarray GetMapping8x();
  bn::ndarray GetTestArray();

private:
  float *pts_d_;
  float *bev_d_;
  float *tr_d_;
  float *premapping1x_d_;
  float *premapping2x_d_;
  float *premapping4x_d_;
  float *premapping8x_d_;
  float *mapping1x_d_;
  float *mapping2x_d_;
  float *mapping4x_d_;
  float *mapping8x_d_;


  float *bev_h_;
  float *mapping1x_h_;
  float *mapping2x_h_;
  float *mapping4x_h_;
  float *mapping8x_h_;
  float *test_array_h_;

  MemorySize memory_size_;
  PreprocessParams params_;
  Timer timer;

  void FreeMemory();
  void MemoryReset(const MemorySize &mz);
  void MemoryAlloc(const MemorySize &mz);
  void FreeMemoryOnDevice();
  void GetArraybnytesSize();
  size_t GetPtsMemorySize(size_t pts_size, size_t pts_num);
  void CopyDataFromDeviceToHost(const MemorySize &mz);
  void CopyDataFromHostToDevice(const bn::ndarray &pts, const bn::ndarray &tr, 
                       const MemorySize &mz, size_t pts_size, size_t pts_num);

  void TestArrayMemoryReset(int bts);

  void TestArrayMemoryAlloc(int bts);

  void CopyTestArrayDataFromDeviceToHost(float *array_h, float *arrray_d, int bts);


};


#endif //_PREPROCESSOR_H
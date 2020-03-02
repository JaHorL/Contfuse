
#ifndef _DATATYPES_H
#define _DATATYPES_H

#include <stdio.h>
#include <cmath>
#include <stdlib.h>

inline void gassert(cudaError_t err_code, const char *file, int line)
{
	if (err_code != cudaSuccess) {
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(err_code), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}
#define CheckCudaErrors(err_code) gassert(err_code, __FILE__, __LINE__);

struct PreprocessParams {

  PreprocessParams() = default;
  PreprocessParams(float x_min, float x_max, float y_min, float y_max, float z_min,
                  float z_max, float x_resolution, float y_resolution, float z_resolution, 
                  size_t sat_z, size_t h, size_t w, float h_scale, float w_scale) : 
                  bev_x_min(x_min), bev_x_max(x_max), 
                  bev_y_min(y_min), bev_y_max(y_max), 
                  bev_z_min(z_min), bev_z_max(z_max), 
                  bev_x_resolution(x_resolution), bev_y_resolution(y_resolution), 
                  bev_z_resolution(z_resolution), bev_sat_z(sat_z), 
                  resized_img_h(h), resized_img_w(w), img_h_scale(h_scale), img_w_scale(w_scale) {
    bev_input_x = ceil((bev_x_max - bev_x_min) / bev_x_resolution - epsilon); 
    bev_input_y = ceil((bev_y_max - bev_y_min) / bev_y_resolution - epsilon); 
    bev_layered_dim = ceil((bev_z_max - bev_z_min) / bev_z_resolution - epsilon); 
    bev_input_z = bev_sat_z + bev_layered_dim;
  }

  float epsilon = 0.0001;
  size_t max_pts_num = 200000;
  size_t premapping_z_d = 3;
  size_t mapping_z_h = 2;
  size_t pt_size = 4;
  size_t tr_size = 4;
  float bev_x_min;
  float bev_x_max;
  float bev_y_min;
  float bev_y_max;
  float bev_z_min;
  float bev_z_max;
  float bev_x_resolution;
  float bev_y_resolution;
  float bev_z_resolution;
  size_t bev_sat_z;
  size_t resized_img_w;
  size_t resized_img_h; 
  float  img_w_scale;
  float  img_h_scale; 
  size_t bev_input_x;
  size_t bev_input_y;
  size_t bev_input_z;
  size_t bev_layered_dim;

};

struct MemorySize {
  MemorySize() = default;
  MemorySize(const PreprocessParams &p) {

    max_pts_bts = sizeof(float) * p.max_pts_num * p.pt_size;
    bev_bts = sizeof(float) * p.bev_input_x * p.bev_input_y * p.bev_input_z;
    tr_bts = sizeof(float) * p.tr_size * p.tr_size;
    premapping1x_d_bts = sizeof(float) * p.bev_input_x * p.bev_input_y * p.premapping_z_d;
    premapping2x_d_bts = premapping1x_d_bts / 4;
    premapping4x_d_bts = premapping1x_d_bts / 16;
    premapping8x_d_bts = premapping1x_d_bts / 64;
    mapping1x_bts = sizeof(float) * p.bev_input_x * p.bev_input_y * p.mapping_z_h;
    mapping2x_bts = mapping1x_bts / 4;
    mapping4x_bts = mapping1x_bts / 16;
    mapping8x_bts = mapping1x_bts / 64;
  }
  size_t max_pts_bts ;
  size_t bev_bts;
  size_t tr_bts;
  size_t premapping1x_d_bts; 
  size_t premapping2x_d_bts;
  size_t premapping4x_d_bts; 
  size_t premapping8x_d_bts;
  size_t mapping1x_bts; 
  size_t mapping2x_bts;
  size_t mapping4x_bts; 
  size_t mapping8x_bts;
};

#endif
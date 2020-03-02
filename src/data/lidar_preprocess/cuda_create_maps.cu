#include "cuda_create_maps.h"

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void CreatePreBevMapOnGPU(float *bev_d, const float *pts_d,
                                     const int pts_num,  PreprocessParams params) {

	int col = blockIdx.x * blockDim.x + threadIdx.x; 
	if(col<pts_num-1)  {
		float pt_x = pts_d[4*col+0];
		float pt_y = pts_d[4*col+1];
		float pt_z = pts_d[4*col+2];
		float pt_i = pts_d[4*col+3];
		if(pt_x>params.bev_x_min && pt_x<params.bev_x_max && pt_y>params.bev_y_min && 
			 pt_y<params.bev_y_max && pt_z>params.bev_z_min && pt_z<params.bev_z_max) {
			int x = (pt_x-params.bev_x_min) / params.bev_x_resolution + 1;
			int y = (pt_y-params.bev_y_min) / params.bev_y_resolution + 1;
			int z = (pt_z-params.bev_z_min) / params.bev_z_resolution;
			int bev_idx = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
																		params.bev_input_z *  (params.bev_input_y - y) + z;
			float point_height = pt_z - params.bev_z_min;
			float point_height_norm = point_height / params.bev_z_resolution - z;
			float old_0 = atomicMax(bev_d+bev_idx, point_height_norm);
			__syncthreads();
			
			int bev_idx_0 = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
									params.bev_input_z *  (params.bev_input_y - y)+ params.bev_layered_dim;
			float old = atomicAdd(bev_d+bev_idx_0, 1);
			__syncthreads();
	
			int bev_idx_1 = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
							params.bev_input_z *  (params.bev_input_y - y) + params.bev_layered_dim + 1;
			old = atomicAdd(bev_d+bev_idx_1, point_height);
			__syncthreads();
	
			int bev_idx_2 = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
							params.bev_input_z *  (params.bev_input_y - y) + params.bev_layered_dim + 2;
			float mean_height = bev_d[bev_idx_1] / bev_d[bev_idx_0];
			float var = (point_height - mean_height)*(point_height - mean_height);
			old = atomicAdd(bev_d+bev_idx_2, var);
			__syncthreads();
			
			int bev_idx_3 = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
							params.bev_input_z * (params.bev_input_y - y) + params.bev_layered_dim + 3;
			int bev_idx_4 = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
							params.bev_input_z * (params.bev_input_y - y) + params.bev_layered_dim + 4;
			int bev_idx_5 = params.bev_input_y * params.bev_input_z * (params.bev_input_x - x) + 
						  params.bev_input_z * (params.bev_input_y - y) + params.bev_layered_dim + 5;
			old = atomicMax(bev_d+bev_idx_4, pt_i);
			if(point_height > bev_d[bev_idx_5]) {
				old = atomicExch(bev_d+bev_idx_3, pt_i);
				old = atomicExch(bev_d+bev_idx_5, point_height);
			}
			__syncthreads();
			}
	}
}

__global__ void CreatePreFusionIdxMapOnGPU(float *mapping1x_d, float *mapping2x_d, 
                                           float *mapping4x_d, float *mapping8x_d, 
																					 const float *pts_d, const int pts_num,
																					 PreprocessParams params) {                                                                                    
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(col<pts_num-1)  {
		float pt_x = pts_d[4*col+0];
		float pt_y = pts_d[4*col+1];
		float pt_z = pts_d[4*col+2];

		if(pt_x>params.bev_x_min && pt_x<params.bev_x_max && pt_y>params.bev_y_min && 
		   pt_y<params.bev_y_max && pt_z>params.bev_z_min && pt_z<params.bev_z_max) {
			// mapping1x_d
			int x1 = (pt_x-params.bev_x_min) / params.bev_x_resolution + 1;
			int y1 = (pt_y-params.bev_y_min) / params.bev_y_resolution + 1;
			int bev_idx1 = params.bev_input_y * params.premapping_z_d * (params.bev_input_x - x1) 
												+ params.premapping_z_d * (params.bev_input_y - y1);
			float point_height = pt_z - params.bev_z_min;
			if(point_height > mapping1x_d[bev_idx1+2]) {
					float old = atomicExch(mapping1x_d+bev_idx1+0, pt_x);
					old = atomicExch(mapping1x_d+bev_idx1+1, pt_y);
					old = atomicExch(mapping1x_d+bev_idx1+2, point_height);
			} 
			// mapping2x_d
			int x2 = (pt_x-params.bev_x_min) / (params.bev_x_resolution*2) + 1;
			int y2 = (pt_y-params.bev_y_min) / (params.bev_y_resolution*2) + 1;
			int bev_idx2 = params.bev_input_y * params.premapping_z_d * (params.bev_input_x/2 - x2) / 2 
												 + params.premapping_z_d * (params.bev_input_y/2 - y2);
			
			if(point_height > mapping2x_d[bev_idx2+2]) {
					float old = atomicExch(mapping2x_d+bev_idx2+0, pt_x);
					old = atomicExch(mapping2x_d+bev_idx2+1, pt_y);
					old = atomicExch(mapping2x_d+bev_idx2+2, point_height);
			} 
			// mapping4x_d
			int x4 = (pt_x-params.bev_x_min) / (params.bev_x_resolution*4) + 1;
			int y4 = (pt_y-params.bev_y_min) / (params.bev_y_resolution*4) + 1;
			int bev_idx4 = params.bev_input_y * params.premapping_z_d * (params.bev_input_x/4 - x4) / 4 
												 + params.premapping_z_d * (params.bev_input_y/4 - y4);
			if(point_height > mapping4x_d[bev_idx4+2]) {
					float old = atomicExch(mapping4x_d+bev_idx4+0, pt_x);
					old = atomicExch(mapping4x_d+bev_idx4+1, pt_y);
					old = atomicExch(mapping4x_d+bev_idx4+2, point_height);
			} 
			// mapping8x_d
			int x8 = (pt_x-params.bev_x_min) / (params.bev_x_resolution*8) + 1;
			int y8 = (pt_y-params.bev_y_min) / (params.bev_y_resolution*8) + 1;
			int bev_idx8 = params.bev_input_y * params.premapping_z_d * (params.bev_input_x/8 - x8) / 8 
												 + params.premapping_z_d * (params.bev_input_y/8 - y8);
			if(point_height > mapping8x_d[bev_idx8+2]) {
					float old = atomicExch(mapping8x_d+bev_idx8+0, pt_x);
					old = atomicExch(mapping8x_d+bev_idx8+1, pt_y);
					old = atomicExch(mapping8x_d+bev_idx8+2, point_height);
			} 
			
		} 
		__syncthreads();
	}
}

__global__ void CreateBevMapOnGPU(float *bev_flip_d, PreprocessParams params) {
	int density_map_idx = params.bev_layered_dim + 0;
	int var_map_idx = params.bev_layered_dim + 2;
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < params.bev_input_y && col < params.bev_input_x) {
			int idx = row * params.bev_input_y * params.bev_input_z + col * params.bev_input_z; 
			if(bev_flip_d[idx+var_map_idx]>2) {
				bev_flip_d[idx+var_map_idx] = 2;
			}
			__syncthreads();
			float density = log(bev_flip_d[idx+density_map_idx]+1) / log(32.0);
			if(density < 1) {
				bev_flip_d[idx+density_map_idx] = density;
			} else {
				bev_flip_d[idx+density_map_idx] = 1;
			}
			__syncthreads();
	}
}

__global__ void CreateFusionIdxMapOnGPU(float *pre_mapping, float *mapping, const float *tr, 
                                        const int downsample_ratio,  PreprocessParams params) { 
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int x_size = params.bev_input_x / downsample_ratio;
	int y_size = params.bev_input_y / downsample_ratio;
	int premapping_z_d = params.premapping_z_d;
	int mapping_z_h = params.mapping_z_h;
	int tr_size = params.tr_size;
	int idx0 = row * y_size * premapping_z_d + col * premapping_z_d;
	int idx1 = row * y_size * mapping_z_h + col * mapping_z_h;
	if(row < y_size && col < x_size) {
			float sum[4] = {0.0f,0.0f,0.0f,0.0f};
			for(int i=0; i<tr_size; i++) {
				if(pre_mapping[idx0+0]>0.1f){
					sum[i] = pre_mapping[idx0+0] * tr[i*tr_size+0] + pre_mapping[idx0+1] * tr[i*tr_size+1];
					sum[i] += (pre_mapping[idx0+2]+params.bev_z_min) * tr[i*tr_size+2] + 1 * tr[i*tr_size+3];
				}
			}
			__syncthreads();
			float x = sum[0]  * params.img_w_scale / (sum[2]*downsample_ratio+params.epsilon);
			float y = sum[1]  * params.img_h_scale / (sum[2]*downsample_ratio+params.epsilon);
			if(y>0 && y< params.resized_img_h / downsample_ratio && x > 0
				 && x < params.resized_img_w / downsample_ratio) {
				mapping[idx1+0] = x;
				mapping[idx1+1] = y;
			}
	} 
}      
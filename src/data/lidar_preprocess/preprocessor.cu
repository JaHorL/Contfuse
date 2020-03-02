#include "preprocessor.h"



double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

size_t Preprocessor::GetPtsMemorySize(size_t pts_size, size_t pts_num) {
	assert(pts_size > 0 && pts_num > 0);
	size_t pts_bts = sizeof(float) * pts_num * pts_size;
	return pts_bts;
}

void Preprocessor::FreeMemory() {
	CheckCudaErrors(cudaFree(pts_d_)); 
	CheckCudaErrors(cudaFree(bev_d_));
	CheckCudaErrors(cudaFree(tr_d_));
	CheckCudaErrors(cudaFree(premapping1x_d_));
	CheckCudaErrors(cudaFree(premapping2x_d_));
	CheckCudaErrors(cudaFree(premapping4x_d_));
	CheckCudaErrors(cudaFree(premapping8x_d_));
	CheckCudaErrors(cudaFree(mapping1x_d_));
	CheckCudaErrors(cudaFree(mapping2x_d_));
	CheckCudaErrors(cudaFree(mapping4x_d_));
	CheckCudaErrors(cudaFree(mapping8x_d_));
	free(bev_h_);
	free(mapping1x_h_);
	free(mapping2x_h_);
	free(mapping4x_h_);
	free(mapping8x_h_);
}

void Preprocessor::MemoryReset(const MemorySize &mz) {
	CheckCudaErrors(cudaMemset(pts_d_, 0.0f, mz.max_pts_bts));
	CheckCudaErrors(cudaMemset(bev_d_, 0.0f, mz.bev_bts));
	CheckCudaErrors(cudaMemset(tr_d_, 0.0f, mz.tr_bts));
	CheckCudaErrors(cudaMemset(mapping1x_d_, 0.0f, mz.mapping1x_bts));
	CheckCudaErrors(cudaMemset(mapping2x_d_, 0.0f, mz.mapping2x_bts));
	CheckCudaErrors(cudaMemset(mapping4x_d_, 0.0f, mz.mapping4x_bts));
	CheckCudaErrors(cudaMemset(mapping8x_d_, 0.0f, mz.mapping8x_bts));
	CheckCudaErrors(cudaMemset(premapping1x_d_, 0.0f, mz.premapping1x_d_bts));
	CheckCudaErrors(cudaMemset(premapping2x_d_, 0.0f, mz.premapping2x_d_bts));
	CheckCudaErrors(cudaMemset(premapping4x_d_, 0.0f, mz.premapping4x_d_bts));
	CheckCudaErrors(cudaMemset(premapping8x_d_, 0.0f, mz.premapping8x_d_bts));
	memset(bev_h_, 0.0f, mz.bev_bts);
	memset(mapping1x_h_, 0.0f, mz.mapping1x_bts);
	memset(mapping2x_h_, 0.0f, mz.mapping2x_bts);
	memset(mapping4x_h_, 0.0f, mz.mapping4x_bts);
	memset(mapping8x_h_, 0.0f, mz.mapping8x_bts);
}


void Preprocessor::MemoryAlloc(const MemorySize &mz) {
	int nb_devices;
	CheckCudaErrors(cudaGetDeviceCount(&nb_devices));
	CheckCudaErrors(cudaSetDevice(0));
	CheckCudaErrors(cudaMalloc((float **) &mapping1x_d_, mz.mapping1x_bts));
	CheckCudaErrors(cudaMalloc((float **) &mapping2x_d_, mz.mapping2x_bts));
	CheckCudaErrors(cudaMalloc((float **) &mapping4x_d_, mz.mapping4x_bts));
	CheckCudaErrors(cudaMalloc((float **) &mapping8x_d_, mz.mapping8x_bts));
	CheckCudaErrors(cudaMalloc((float **) &premapping1x_d_, mz.premapping1x_d_bts));
	CheckCudaErrors(cudaMalloc((float **) &premapping2x_d_, mz.premapping2x_d_bts));
	CheckCudaErrors(cudaMalloc((float **) &premapping4x_d_, mz.premapping4x_d_bts));
	CheckCudaErrors(cudaMalloc((float **) &premapping8x_d_, mz.premapping8x_d_bts));
	CheckCudaErrors(cudaMalloc((float **) &pts_d_, mz.max_pts_bts));
	CheckCudaErrors(cudaMalloc((float **) &bev_d_, mz.bev_bts));
	CheckCudaErrors(cudaMalloc((float **) &tr_d_, mz.tr_bts));
	bev_h_ = (float*) malloc(mz.bev_bts);
	mapping1x_h_ = (float*) malloc(mz.mapping1x_bts);
	mapping2x_h_ = (float*) malloc(mz.mapping2x_bts);
	mapping4x_h_ = (float*) malloc(mz.mapping4x_bts);
	mapping8x_h_ = (float*) malloc(mz.mapping8x_bts);
}



void Preprocessor::CopyDataFromHostToDevice(const bn::ndarray &pts, 
													const bn::ndarray &tr, const MemorySize &mz, 
													size_t pts_size, size_t pts_num) {
	size_t pts_bts = GetPtsMemorySize(pts_size, pts_num);
	CheckCudaErrors(cudaMemcpy(pts_d_, pts.get_data(), pts_bts, cudaMemcpyHostToDevice));
	CheckCudaErrors(cudaMemcpy(tr_d_, tr.get_data(), mz.tr_bts, cudaMemcpyHostToDevice));
}


void Preprocessor::CopyDataFromDeviceToHost(const MemorySize &mz) {
	CheckCudaErrors(cudaMemcpy(bev_h_, bev_d_, mz.bev_bts, cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaMemcpy(mapping1x_h_, mapping1x_d_, mz.mapping1x_bts, 
														 cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaMemcpy(mapping2x_h_, mapping2x_d_, mz.mapping2x_bts, 
														 cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaMemcpy(mapping4x_h_, mapping4x_d_, mz.mapping4x_bts, 
														 cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaMemcpy(mapping8x_h_, mapping8x_d_, mz.mapping8x_bts, 
														 cudaMemcpyDeviceToHost));
}

void Preprocessor::TestArrayMemoryReset(int bts) {
	memset(test_array_h_, 0.0f, bts);
}

void Preprocessor::TestArrayMemoryAlloc(int bts) {
	test_array_h_ = (float *) malloc(bts);
}

void Preprocessor::CopyTestArrayDataFromDeviceToHost(float *array_h, float *arrray_d, int bts) {
	CheckCudaErrors(cudaMemcpy(array_h, arrray_d, bts, cudaMemcpyDeviceToHost)); 
}

void Preprocessor::PreprocessorInit(float bev_x_min,        float bev_x_max,
																				float bev_y_min,        float bev_y_max,
																				float bev_z_min,        float bev_z_max,
																				float bev_x_resolution, float bev_y_resolution,
																				float bev_z_resolution, size_t bev_sat_z,
																				size_t h,               size_t w,
																			  float h_scale,          float w_scale) {
	Py_Initialize();
	bn::initialize();
	params_ = PreprocessParams(bev_x_min, bev_x_max, bev_y_min, bev_y_max,
														 bev_z_min, bev_z_max, bev_x_resolution, bev_y_resolution,
														 bev_z_resolution, bev_sat_z, h, w, h_scale, w_scale);
	memory_size_ = MemorySize(params_);
	ParamsPrint(params_, memory_size_);
	MemoryAlloc(memory_size_);
	// TestArrayMemoryAlloc(memory_size_.premapping1x_d_bts);
}

void Preprocessor::PreprocessData(const bn::ndarray &lidar, const bn::ndarray &tr, size_t pts_num) {
	// timer.start();
	MemoryReset(memory_size_);
	// TestArrayMemoryReset(memory_size_.premapping1x_d_bts);
	CopyDataFromHostToDevice(lidar, tr, memory_size_,  params_.pt_size, pts_num);
	// printf("CopyDataFromHostToDevice: %4f.\n", timer.stop());
	dim3 grid_0(256, 1, 1);
	dim3 block_0(768, 1, 1);
	CreatePreBevMapOnGPU<<<grid_0, block_0>>>(bev_d_, pts_d_, pts_num, params_);
	CheckCudaErrors(cudaDeviceSynchronize());
	CheckCudaErrors(cudaGetLastError());
	// printf("CreatePreBevMapOnGPU: %4f.\n", timer.stop());
	CreatePreFusionIdxMapOnGPU<<<grid_0, block_0>>>(premapping1x_d_, premapping2x_d_, 
																									premapping4x_d_, premapping8x_d_, 
																									pts_d_, 				 pts_num,
																								  params_);
	CheckCudaErrors(cudaDeviceSynchronize());
	CheckCudaErrors(cudaGetLastError());
	// printf("CreatePreFusionIdxMapOnGPU: %4f.\n", timer.stop());
	dim3 grid_1(32, 32);
	dim3 block_1(32, 32);
	CreateBevMapOnGPU<<<grid_1, block_1>>>(bev_d_, params_);
	CheckCudaErrors(cudaDeviceSynchronize());
	CheckCudaErrors(cudaGetLastError());
	// printf("CreateBevMapOnGPU: %4f.\n", timer.stop());
	dim3 grid_2(16,16);
	dim3 grid_3(8,8);
	dim3 grid_4(4,4);
	CreateFusionIdxMapOnGPU<<<grid_1, block_1>>>(premapping1x_d_, mapping1x_d_, tr_d_, 1, params_);
	CreateFusionIdxMapOnGPU<<<grid_2, block_1>>>(premapping2x_d_, mapping2x_d_, tr_d_, 2, params_);
	CreateFusionIdxMapOnGPU<<<grid_3, block_1>>>(premapping4x_d_, mapping4x_d_, tr_d_, 4, params_);
	CreateFusionIdxMapOnGPU<<<grid_4, block_1>>>(premapping8x_d_, mapping8x_d_, tr_d_, 8, params_);
	CheckCudaErrors(cudaDeviceSynchronize());
	CheckCudaErrors(cudaGetLastError());
	// printf("CreateFusionIdxMapOnGPU: %4f.\n", timer.stop());
	CopyDataFromDeviceToHost(memory_size_);	
	// CopyTestArrayDataFromDeviceToHost(test_array_h_, premapping1x_d_, memory_size_.premapping1x_d_bts);
	CheckCudaErrors(cudaDeviceSynchronize());
	CheckCudaErrors(cudaGetLastError());		
	// printf("CopyDataFromDeviceToHost: %4f \n.", timer.stop());
}

bn::ndarray Preprocessor::GetBev() {
	bp::tuple shape = bp::make_tuple(params_.bev_input_x, params_.bev_input_y, params_.bev_input_z);
	bp::tuple stride = bp::make_tuple(params_.bev_input_y * params_.bev_input_z * sizeof(float),
																	  params_.bev_input_z * sizeof(float),  sizeof(float));
	bn::dtype dt1 = bn::dtype::get_builtin<float>();
	return bn::from_data(bev_h_, dt1, shape, stride, bp::object());
}

bn::ndarray Preprocessor::GetMapping1x() {
	bp::tuple shape = bp::make_tuple(params_.bev_input_x, params_.bev_input_y, params_.mapping_z_h);
	bp::tuple stride = bp::make_tuple(params_.bev_input_y * params_.mapping_z_h * sizeof(float), 
																		params_.mapping_z_h * sizeof(float) , sizeof(float));
	bn::dtype dt1 = bn::dtype::get_builtin<float>();
	return bn::from_data(mapping1x_h_, dt1, shape, stride, bp::object());
}

bn::ndarray Preprocessor::GetMapping2x() {
	bp::tuple shape = bp::make_tuple(params_.bev_input_x/2, params_.bev_input_y/2, params_.mapping_z_h);
	bp::tuple stride = bp::make_tuple(params_.bev_input_y * params_.mapping_z_h * sizeof(float) / 2, 
																		params_.mapping_z_h * sizeof(float), sizeof(float));
	bn::dtype dt1 = bn::dtype::get_builtin<float>();
	return bn::from_data(mapping2x_h_, dt1, shape, stride, bp::object());
}

bn::ndarray Preprocessor::GetMapping4x() {
	bp::tuple shape = bp::make_tuple(params_.bev_input_x/4, params_.bev_input_y/4, params_.mapping_z_h);
	bp::tuple stride = bp::make_tuple(params_.bev_input_y * params_.mapping_z_h * sizeof(float) / 4, 
																		params_.mapping_z_h * sizeof(float), sizeof(float));
	bn::dtype dt1 = bn::dtype::get_builtin<float>();
	return bn::from_data(mapping4x_h_, dt1, shape, stride, bp::object());
}

bn::ndarray Preprocessor::GetMapping8x() {
	bp::tuple shape = bp::make_tuple(params_.bev_input_x/8, params_.bev_input_y/8, params_.mapping_z_h);
	bp::tuple stride = bp::make_tuple(params_.bev_input_y * params_.mapping_z_h * sizeof(float) / 8, 
																		params_.mapping_z_h * sizeof(float), sizeof(float));
	bn::dtype dt1 = bn::dtype::get_builtin<float>();
	return bn::from_data(mapping8x_h_, dt1, shape, stride, bp::object());
}

bn::ndarray Preprocessor::GetTestArray() {
	bp::tuple shape = bp::make_tuple(params_.bev_input_x, params_.bev_input_y, params_.premapping_z_d);
	bp::tuple stride = bp::make_tuple(params_.bev_input_y * params_.premapping_z_d * sizeof(float), 
																		params_.premapping_z_d * sizeof(float) , sizeof(float));
	bn::dtype dt1 = bn::dtype::get_builtin<float>();
	return bn::from_data(test_array_h_, dt1, shape, stride, bp::object());
}

BOOST_PYTHON_MODULE(libcuda_preprocessor) {
	Py_Initialize();
	bn::initialize();
	bp::class_<Preprocessor>("Preprocessor")
				.def("PreprocessorInit", &Preprocessor::PreprocessorInit)
        .def("PreprocessData", &Preprocessor::PreprocessData)
        .def("GetMapping1x", &Preprocessor::GetMapping1x)
        .def("GetMapping2x", &Preprocessor::GetMapping2x)
        .def("GetMapping4x", &Preprocessor::GetMapping4x)
        .def("GetMapping8x", &Preprocessor::GetMapping8x)
				// .def("GetTestArray", &Preprocessor::GetTestArray)
        .def("GetBev", &Preprocessor::GetBev);
}


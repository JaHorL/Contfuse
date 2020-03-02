#include "debug_utils.h"

void PrintArray(const float* array, int size) {
	printf("============ %d ===========", size);
	for(int i=0; i<size; ++i) {
		printf("%f ", array[i]);
	}
}

void ParamsPrint(const PreprocessParams &params, const MemorySize &mz) {
	std::cout << "--------------------------" << std::endl;
	std::cout << "--------------------------" << std::endl;
	std::cout << "Print Preprocess Params:" << std::endl;
	std::cout << "bev xyz range:" << std::endl;
	std::cout << params.bev_x_min << " " << params.bev_x_max << " " 
						<< params.bev_y_min << " " << params.bev_y_max << " "
						<< params.bev_z_min << " " << params.bev_z_max << " "
						<< std::endl;
	std::cout << "bev resolution:" << std::endl;
	std::cout << params.bev_x_resolution << " " 
						<< params.bev_y_resolution << " "
						<< params.bev_z_resolution << " "
						<< std::endl;
	std::cout << "bev input size:" << std::endl;
	std::cout << params.bev_input_x << " " 
						<< params.bev_input_y << " "
						<< params.bev_input_z << " "
						<< std::endl;
	std::cout << "img input size:" << std::endl;
	std::cout << params.resized_img_h << " " 
						<< params.resized_img_w << " "
						<< params.img_h_scale << " "
						<< params.img_w_scale << " "
						<< std::endl;
	std::cout << "bev_sat_z: " << params.bev_sat_z <<"\n"
						<< "bev_layered_dim: " << params.bev_layered_dim<<"\n"
						<< "tr_size: " << params.tr_size << "\n"
						<< "pt_size: " << params.pt_size << "\n" 
						<< "mapping_z_h: " << params.mapping_z_h << "\n"
						<< "premapping_z_d: " << params.premapping_z_d << "\n" 
						<< "epsilon: " << params.epsilon << std::endl; 
	std::cout << "--------------------------" << std::endl;
	std::cout << "--------------------------" << std::endl;
	std::cout << "Print MemorySize Params:" << std::endl;
	std::cout << "max_pts_bts: " << mz.max_pts_bts << "\n"
						<< "bev_bts: "  << mz.bev_bts << "\n"
						<< "tr_bts: " << mz.tr_bts << std::endl;
	std::cout << "premapping1x_d_bts: " << mz.premapping1x_d_bts << "\n"
						<< "mapping1x_bts: " << mz.mapping1x_bts << std::endl;
}



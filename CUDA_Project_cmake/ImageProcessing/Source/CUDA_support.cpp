#include<CUDA_support.h>

void cudaCheckErrors(cudaError_t err) {
	do {
			err = cudaGetLastError(); 
			if (err != cudaSuccess) {
					fprintf(stderr, "Fatal error: %s (%s:%d)\n",cudaGetErrorString(err),__FILE__, __LINE__); 
					fprintf(stderr, "*** FAILED - ABORTING\n"); 
			} 
	} while (0);
}

namespace CUDA {
	void initCUDADevice() {
		//CUDA code
		//get the number of GPU availble in PC
		int numDevices;
		int device;
		cudaCheckErrors(cudaGetDeviceCount(&numDevices));
		cudaCheckErrors(cudaGetDevice(&device));
		cudaDeviceProp gpuProps;
		cudaGetDeviceProperties(&gpuProps, device);
		cudaSetDevice(device);
	}
}
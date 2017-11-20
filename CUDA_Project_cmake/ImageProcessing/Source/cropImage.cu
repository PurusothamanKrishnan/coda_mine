#include"common_structs.h"
//#include"imageOperations.h"
#include<stdio.h>
#include<iostream>
#define BLOCK_WIDTH 10
#define BLOCK_HEIGHT 60

using namespace std;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define cudaCheckErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
__global__ void cropImageFunctionCuda(tUInt_8* input, tUInt_8* output, tUInt_32 InputPitch, tUInt_32 outputPitch, tUInt_32 offsetX, tUInt_32 offsetY,tUInt_32 channels) {

	tUInt_32 idxValX = blockDim.x * blockIdx.x + threadIdx.x;
	tUInt_32 idxValY = blockDim.y * blockIdx.y + threadIdx.y;
	
	tUInt_32 inputIdx = (idxValY + offsetY) * InputPitch + (idxValX + offsetX) * channels;
	tUInt_32 outputIdx = idxValY  * outputPitch + idxValX * channels;

	output[outputIdx] = input[inputIdx];
	output[outputIdx + 1] = input[inputIdx + 1];
	output[outputIdx + 2] = input[inputIdx + 2];
}


extern void cropImageFunctionHost(tUInt_8* inputBuffer, tUInt_8* outputBuffer, IMAGE_INFO inputImageInfo, IMAGE_INFO outputImageInfo, tUInt_32 offsetX, tUInt_32 offsetY, tUInt_8** d_cropOutImage) {
	tUInt_8* d_inputImage;
	size_t sizeInput = inputImageInfo.width * inputImageInfo.height * inputImageInfo.channnels * sizeof(tUInt_8);
	size_t sizeOutput = outputImageInfo.width * outputImageInfo.height * outputImageInfo.channnels * sizeof(tUInt_8);
	tUInt_32 inputPitch = inputImageInfo.width * inputImageInfo.channnels;
	tUInt_32 outputPitch = outputImageInfo.width * outputImageInfo.channnels;
	cudaCheckErrors(cudaMalloc(&d_inputImage, sizeInput));
	cudaCheckErrors(cudaMalloc(d_cropOutImage, sizeOutput));

	cudaCheckErrors(cudaMemcpy(d_inputImage, inputBuffer, sizeInput, cudaMemcpyHostToDevice));


	dim3 threadsPerBlock(10, 6);
	dim3 numBlocks(outputImageInfo.width / threadsPerBlock.x, outputImageInfo.height / threadsPerBlock.y);
	cropImageFunctionCuda<<<numBlocks, threadsPerBlock, 0>> >(d_inputImage, *d_cropOutImage, inputPitch, outputPitch, offsetX, offsetY, inputImageInfo.channnels);

	cudaCheckErrors(cudaMemcpy(outputBuffer, *d_cropOutImage, sizeOutput, cudaMemcpyDeviceToHost));

	cudaCheckErrors(cudaFree(d_inputImage));
}

void temp_func() {
cout<<"SUccccesssssssssssssssssssssssssssssssss"<<endl;
// temp code
}

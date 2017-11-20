#include"common_structs.h"
#include<stdio.h>
#include"imageOperations.h"
#define BLOCK_WIDTH 10
#define BLOCK_HEIGHT 60


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
__global__ void rgb2GreyImage(tUInt_8* input, tUInt_8* output, tUInt_32 InputPitch, tUInt_32 outputPitch, tUInt_32 inputChannels,tUInt_32 outputChannels) {

	tUInt_32 idxValX = blockDim.x * blockIdx.x + threadIdx.x;
	tUInt_32 idxValY = blockDim.y * blockIdx.y + threadIdx.y;
	
	tUInt_32 inputIdx = idxValY * InputPitch + idxValX * inputChannels;
	tUInt_32 outputIdx = idxValY  * outputPitch + idxValX * outputChannels;

	output[outputIdx] = (tUInt_8) (0.2989 * input[inputIdx + 2] + 0.5870 * input[inputIdx + 1] + 0.1140 * input[inputIdx]);
}


extern void rgb2GreyImageFunctionHost(tUInt_8* d_inputBuffer, tUInt_8* outputBuffer, IMAGE_INFO inputImageInfo, IMAGE_INFO outputImageInfo,tUInt_8** d_greyImage) {
	size_t sizeInput = inputImageInfo.width * inputImageInfo.height * inputImageInfo.channnels * sizeof(tUInt_8);
	size_t sizeOutput = outputImageInfo.width * outputImageInfo.height * outputImageInfo.channnels * sizeof(tUInt_8);
	tUInt_32 inputPitch = inputImageInfo.width * inputImageInfo.channnels;
	tUInt_32 outputPitch = outputImageInfo.width * outputImageInfo.channnels;
	
	cudaCheckErrors(cudaMalloc(d_greyImage, sizeOutput));

	dim3 threadsPerBlock(10, 6); 
	dim3 numBlocks(outputImageInfo.width / threadsPerBlock.x, outputImageInfo.height / threadsPerBlock.y);
	rgb2GreyImage<<<numBlocks,threadsPerBlock>>>(d_inputBuffer,*d_greyImage,inputPitch,outputPitch,inputImageInfo.channnels,outputImageInfo.channnels);
	
	cudaCheckErrors(cudaMemcpy(outputBuffer,*d_greyImage,sizeOutput,cudaMemcpyDeviceToHost));
}

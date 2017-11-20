#include"common_structs.h"
#include<stdio.h>
#include<iostream>
#define BLOCK_WIDTH 10
#define BLOCK_HEIGHT 60
#include"imageOperations.h"
using namespace std;
#define ABS(x) x > 0?x:-x;
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
__global__ void sobelKernal(tUInt_8* input, tUInt_8* output, tUInt_32 rows, tUInt_32 cols, tUInt_32 channels) {

	tUInt_32 idxValX = blockDim.x * blockIdx.x + threadIdx.x;
	tUInt_32 idxValY = blockDim.y * blockIdx.y + threadIdx.y;
	
	tUInt_32 inputIdx = idxValY  * cols * channels + idxValX * channels;
	tUInt_32 outputIdx = idxValY  * cols * channels + idxValX * channels;

	tInt_32 i = idxValY;
	tInt_32 j = idxValX;
	tInt_32 i_start= (i - 1);
	tInt_32 j_start= (j - 1);
	
	tInt_32 rows_minus_one = rows - 1;
	tInt_32 cols_minus_one = cols - 1;

	tInt_32 i_end= rows_minus_one - (i + 1);
	tInt_32 j_end= cols_minus_one - (j + 1);
	

	i_start = ABS(i_start);
	j_start = ABS(j_start);
	i_end = ABS(i_end);
	j_end = ABS(j_end);
	
	i_end = rows_minus_one - i_end;
	j_end = cols_minus_one - j_end;

	tUInt_8* inputPtrLine0 = input + (i_start * cols * channels);
	tUInt_8* inputPtrLine2 = input + (i_end * cols * channels);
	
	tInt_32 val = (*(inputPtrLine0 + j_start * channels) *  -3) + (*(inputPtrLine0 + j * channels) *  -10) + (*(inputPtrLine0 + j_end * channels) *  -3) + (*(inputPtrLine2 + j_start * channels) * 3) + (*(inputPtrLine2 + j * channels) * 10) + (*(inputPtrLine2 + j_end * channels) * 3);
	output[outputIdx] = (tUInt_8)((val > 1700)?255:0);
}


extern void sobelImageFunctionHost(tUInt_8* d_inputBuffer, tUInt_8* outputBuffer, IMAGE_INFO inputImageInfo, IMAGE_INFO outputImageInfo) {
	tUInt_8* d_sobelImage;
	size_t sizeOutput = outputImageInfo.width * outputImageInfo.height * outputImageInfo.channnels;

	cudaCheckErrors(cudaMalloc(&d_sobelImage, sizeOutput));
	
	dim3 threadsPerBlock(10, 6); 
	dim3 numBlocks(outputImageInfo.width / threadsPerBlock.x, outputImageInfo.height / threadsPerBlock.y);
	sobelKernal<<<numBlocks,threadsPerBlock,0>>>(d_inputBuffer,d_sobelImage,outputImageInfo.height,outputImageInfo.width,outputImageInfo.channnels);
	
	cudaCheckErrors(cudaMemcpy(outputBuffer,d_sobelImage,sizeOutput,cudaMemcpyDeviceToHost));

	cudaCheckErrors(cudaFree(d_sobelImage));
}

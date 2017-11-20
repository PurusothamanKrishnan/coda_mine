#ifndef IMAGE_OPRATIONS_H
#define IMAGE_OPERATION_H
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include"common_structs.h"
#include"CUDA_support.h"
#include<cuda.h>

extern void cropImageWrapper(tUInt_8*, tUInt_8*, IMAGE_INFO, IMAGE_INFO, tUInt_32, tUInt_32, tUInt_8**);
extern void cropImageFunctionHost(tUInt_8*, tUInt_8*, IMAGE_INFO, IMAGE_INFO, tUInt_32, tUInt_32, tUInt_8**);
extern void temp_func();
extern void rgb2GreyImageFunctionHost(tUInt_8* d_inputBuffer, tUInt_8* outputBuffer, IMAGE_INFO inputImageInfo, IMAGE_INFO outputImageInfo, tUInt_8** d_greyImage);
extern void sobelImageFunctionHost(tUInt_8* d_inputBuffer, tUInt_8* outputBuffer, IMAGE_INFO inputImageInfo, IMAGE_INFO outputImageInfo);

#endif
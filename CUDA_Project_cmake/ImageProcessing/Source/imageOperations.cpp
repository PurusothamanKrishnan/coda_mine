#ifndef IMAGE_OPRATIONS_H
#define IMAGE_OPERATION_H
#include"imageOperations.h"

void cropImageWrapper(tUInt_8* inputBuffer, tUInt_8* outputBuffer, IMAGE_INFO inputImageInfo, IMAGE_INFO outputImageInfo, tUInt_32 offsetX, tUInt_32 offsetY, tUInt_8** d_cropOutImage) {
	cropImageFunctionHost(inputBuffer,outputBuffer,inputImageInfo,outputImageInfo,offsetX,offsetY,d_cropOutImage);	
	temp_func();
}

#endif
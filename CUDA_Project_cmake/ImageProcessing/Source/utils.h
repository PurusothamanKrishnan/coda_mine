#ifndef UTILS_H
#define UTILS_H

#include"common_structs.h"
#include"konstantsHost.h"

namespace utils {

	void initAPP(ROI &roi) {
		roi.xOffset = xOffset;
		roi.yOffset = yOffset;
		roi.width = width;
		roi.height = height;
	}
	template <typename T>
	void cropImage(T* outputBuffer, T*inputBuffer, const IMAGE_INFO inputImageInfo, const IMAGE_INFO outputImageInfo, tUInt_32 xOffset, tUInt_32 yOffset) {
		assert(inputImageInfo.channnels == outputImageInfo.channnels);
		assert(inputImageInfo.width >= (outputImageInfo.width + xOffset));
		assert(inputImageInfo.height >= (outputImageInfo.height + xOffset));

		const tUInt_16 outputImagePitch = outputImageInfo.width * outputImageInfo.channnels;
		for (tUInt_16 i = 0; i < outputImageInfo.height; i++) {
			T* inputPtr = inputBuffer + ((i + yOffset) * inputImageInfo.width * inputImageInfo.channnels) + xOffset * inputImageInfo.channnels;
			T* outputPtr = outputBuffer + (i  * outputImageInfo.width * outputImageInfo.channnels);
			for (tUInt_16 j = 0; j < outputImagePitch; j++) {
				outputPtr[j] = *(inputPtr + j);
			}
		}
	}

	template <typename T_Src,typename T_Dst>
	void rgb2grey(T_Dst* outputBuffer, T_Src* inputBuffer, const IMAGE_INFO inputImageInfo, const IMAGE_INFO outputImageInfo) {
		assert(inputImageInfo.channnels != outputImageInfo.channnels);
		assert(inputImageInfo.width  == outputImageInfo.width);
		assert(inputImageInfo.height == outputImageInfo.height);

		for (tUInt_16 i = 0; i < outputImageInfo.height; i++) {
			T_Src* inputPtr = inputBuffer + (i * inputImageInfo.width * inputImageInfo.channnels);
			T_Dst* outputPtr = outputBuffer + (i  * outputImageInfo.width * outputImageInfo.channnels);
			for (tUInt_16 j = 0; j < outputImageInfo.width; j++) {
				*outputPtr = static_cast<T_Dst> (0.2989 * * (inputPtr + 2) + 0.5870 * * (inputPtr + 1) + 0.1140 * *inputPtr);
				inputPtr += inputImageInfo.channnels;
				outputPtr += outputImageInfo.channnels;
			}
		}
	}


	template <typename T_Src, typename T_Dst>
	void sobel(T_Dst* outputBuffer, T_Src* inputBuffer, const IMAGE_INFO inputImageInfo, const IMAGE_INFO outputImageInfo) {
		assert(inputImageInfo.channnels == outputImageInfo.channnels);
		assert(inputImageInfo.width == outputImageInfo.width);
		assert(inputImageInfo.height == outputImageInfo.height);

		for (tUInt_16 i = 0; i < outputImageInfo.height; i++) {
			tInt_32 i_start = abs(i - 1);
			tInt_32 i_end = (outputImageInfo.height - 1) - abs((outputImageInfo.height - 1) - (i + 1));

			T_Src* inputPtrLine0 = inputBuffer + (i_start * inputImageInfo.width * inputImageInfo.channnels);
			T_Src* inputPtrLine1 = inputBuffer + (i * inputImageInfo.width * inputImageInfo.channnels);
			T_Src* inputPtrLine2 = inputBuffer + (i_end * inputImageInfo.width * inputImageInfo.channnels);

			T_Dst* outputPtr = outputBuffer + (i  * outputImageInfo.width * outputImageInfo.channnels);
			for (tUInt_16 j = 0; j < outputImageInfo.width; j++) {

				tInt_32 j_start = abs(j - 1);
				tInt_32 j_end = (outputImageInfo.width - 1) - abs((outputImageInfo.width - 1) - (j + 1));

				tInt_32 val = (*(inputPtrLine0 + j_start * inputImageInfo.channnels) *  -3) + (*(inputPtrLine0 + j * inputImageInfo.channnels) *  -10) + (*(inputPtrLine0 + j_end * inputImageInfo.channnels) *  -3) + (*(inputPtrLine2 + j_start * inputImageInfo.channnels) * 3) + (*(inputPtrLine2 + j * inputImageInfo.channnels) * 10) + (*(inputPtrLine2 + j_end * inputImageInfo.channnels) * 3);
				*(outputPtr + j * outputImageInfo.channnels) = (T_Dst)((val > 1700)?255:0);
			}
		}
	}

}
#endif
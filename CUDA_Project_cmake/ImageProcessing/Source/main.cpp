#include<iostream>
#include<fstream>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\opencv.hpp>
#include"CUDA_support.h"
#include"imageOperations.h"

#include"utils.h"
using namespace std;
using namespace cv;
int main(int argc, char *argv[])
{
	// Mat declartions
	Mat inputImage;
	Mat matCropImage; 
	Mat greyImage;
	Mat sobelImage;
	Mat sobelImageCUDA;
	//ROI declarations
	ROI roi;
	//Image information
	IMAGE_INFO inputImageInfo;
	IMAGE_INFO outputImageInfo;
	IMAGE_INFO greyImageInfo;

	Size size_CropImage;
	utils::initAPP(roi);
	inputImage = imread("D:/images/image.png", CV_LOAD_IMAGE_COLOR);
	/*if (argc < 2) {
		cout << "Insput arguments are missing" << endl;
		return 0;
	}*/
	//init CUDA - Testing 
	CUDA::initCUDADevice();

	inputImageInfo.width = inputImage.cols;
	inputImageInfo.height = inputImage.rows;
	inputImageInfo.channnels = inputImage.channels();

	outputImageInfo.channnels = inputImage.channels();
	outputImageInfo.width = roi.width;
	outputImageInfo.height = roi.height;

	greyImageInfo.channnels = 1;
	greyImageInfo.width = roi.width;
	greyImageInfo.height = roi.height;
	
	size_CropImage.height = roi.height;
	size_CropImage.width = roi.width;

	matCropImage.create(size_CropImage, CV_8UC3);
	greyImage.create(size_CropImage, CV_8UC1);
	sobelImage.create(size_CropImage, CV_8UC1);
	sobelImageCUDA.create(size_CropImage, CV_8UC1);
	tUInt_8* d_outCropImage = NULL;
	tUInt_8* d_greyImage = NULL;
	//utils::cropImage<uchar>(matCropImage.data, inputImage.data, inputImageInfo, outputImageInfo, xOffset, yOffset);
	cropImageFunctionHost(inputImage.data, matCropImage.data, inputImageInfo, outputImageInfo, xOffset, yOffset, &d_outCropImage);
	rgb2GreyImageFunctionHost(d_outCropImage, greyImage.data, outputImageInfo, greyImageInfo, &d_greyImage);
	utils::sobel(sobelImage.data, greyImage.data, greyImageInfo, greyImageInfo);
	//utils::rgb2grey<tUInt_8, tUInt_8>(greyImage.data, matCropImage.data, outputImageInfo, greyImageInfo);
	sobelImageFunctionHost(d_greyImage, sobelImageCUDA.data, greyImageInfo, greyImageInfo);
	imshow("Input Image", inputImage);
	imshow("Cropped Image", matCropImage);
	imshow("grey Image", greyImage);
	imshow("sobelImage", sobelImage);
	imshow("sobelImageCUDA", sobelImageCUDA);
	waitKey(0);
	cout << "App Start" << endl;
	return 1;
}
#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H
#include<cassert>
typedef unsigned long int	tUInt_64;
typedef unsigned int		tUInt_32;
typedef unsigned short		tUInt_16;
typedef unsigned char		tUInt_8;

typedef long int	tInt_64;
typedef int			tInt_32;
typedef short		tInt_16;
typedef char		tInt_8;

typedef float  flt;
typedef double dle;



typedef struct imageInfo {
	tUInt_16 width;
	tUInt_16 height;
	tUInt_16 channnels;
} IMAGE_INFO;


typedef struct roiImage {
	tUInt_16 xOffset;
	tUInt_16 yOffset;
	tUInt_16 width;
	tUInt_16 height;
} ROI;
#endif
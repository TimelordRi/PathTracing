#pragma pack(2)
#include<iostream>

typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned long  DWORD;
typedef long    LONG;

//struct BITMAPFILEHEADER
//{
//    WORD  bfType;		
//    DWORD bfSize;		
//    WORD  bfReserved1;	
//    WORD  bfReserved2;	
//    DWORD bfOffBits;	
//};
//
//
//struct BITMAPINFOHEADER
//{
//    DWORD biSize;			
//    LONG  biWidth;			
//    LONG  biHeight;			
//    WORD  biPlanes;			
//    WORD  biBitCount;		
//    DWORD biCompression;    
//    DWORD biSizeImage;      
//    LONG  biXPelsPerMeter;  
//    LONG  biYPelsPerMeter;  
//    DWORD biClrUsed;        
//    DWORD biClrImportant;	
//};


struct RGBColor
{
    char B;		
    char G;		
    char R;		
};

void WriteBMP(const char* FileName, RGBColor* ColorBuffer, int ImageWidth, int ImageHeight)
{
    
    const int ColorBufferSize = ImageHeight * ImageWidth * sizeof(RGBColor);

    
    BITMAPFILEHEADER fileHeader;
    fileHeader.bfType = 0x4D42;	
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + ColorBufferSize;
    fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    
    BITMAPINFOHEADER bitmapHeader = { 0 };
    bitmapHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapHeader.biHeight = ImageHeight;
    bitmapHeader.biWidth = ImageWidth;
    bitmapHeader.biPlanes = 1;
    bitmapHeader.biBitCount = 24;
    bitmapHeader.biSizeImage = ColorBufferSize;
    bitmapHeader.biCompression = 0; 


    FILE* fp;
    fopen_s(&fp, FileName, "wb");
    fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
    fwrite(&bitmapHeader, sizeof(BITMAPINFOHEADER), 1, fp);
    fwrite(ColorBuffer, ColorBufferSize, 1, fp);
    fclose(fp);
}
// 670010894879
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <iostream>
#include <fstream>
#include <string>

struct Pixel
{
  uint8_t blue;
  uint8_t green;
  uint8_t red;
  uint8_t alpha;
};

struct BMP
{
  struct {
    int16_t header;
    uint32_t filesize;
    int16_t reser;
    int16_t reser1;
    uint32_t dataoffset;
  } header;
  struct {
    uint32_t headersize;
    int32_t width;
    int32_t height;
    uint16_t plans;
    uint16_t bpp;
    uint32_t compression;
    uint32_t datasize;
    int32_t re;
    int32_t ve;
    uint32_t color;
    uint32_t importantcolor;
  } info;
  thrust::device_vector<Pixel> data;
};

BMP readBMP(const std::string& filename)
{
    FILE* f = fopen(filename.c_str(), "rb");

    if(f == NULL)
        throw "Argument Exception";

    BMP img;

    fread(&img.header, 14, 1, f);

    fread(&img.info, 40, 1, f);

    thrust::host_vector<Pixel> temp;
    temp.resize(img.info.width * img.info.height);

    fread(&temp[0], sizeof(Pixel), img.info.width * img.info.height, f);

    img.data = thrust::device_vector<Pixel>(temp.begin(), temp.end());

    fclose(f);

    return img;
}

void writeBMP(const std::string& filename, const BMP& image)
{
    FILE* f = fopen(filename.c_str(), "wb");

    fwrite(&image.header, 14, 1, f);

    fwrite(&image.info, 40, 1, f);

    thrust::host_vector<Pixel> temp(image.data.begin(), image.data.end());
    fwrite(&temp[0], sizeof(Pixel), image.data.size(), f); 

    fclose(f);
}

struct reverse_colors
{
  reverse_colors(){}
  __host__ __device__ Pixel operator()(const Pixel& pixel) const { 
      Pixel _pixel;
      _pixel.green = 255 - pixel.green;
      _pixel.red = 255 - pixel.red;
      _pixel.blue = 255 - pixel.blue;
      _pixel.alpha = 255 - pixel.alpha;
      return _pixel;
  }
};

__global__ void increase_contrast(Pixel* A, Pixel* B, size_t rows, size_t cols)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;

  int i = n / cols;
  int j = n % cols;

  if (i < 0 || j < 0 || i >= rows || j >= cols)
    return;

  if (i > 0 && j > 0 && i < rows - 1 && j < cols - 1)
  {
    const auto val_of_cell = [A, rows, cols](size_t i, size_t j) -> Pixel& {
      return A[i * cols + j];
    };
    
    int green = (int)(val_of_cell(i - 1, j).green) * -1 
               + (int)(val_of_cell(i, j - 1).green) * -1 
               + (int)(val_of_cell(i + 1, j).green) * -1 
               + (int)(val_of_cell(i, j + 1).green) * -1 
               + (int)(val_of_cell(i, j).green) * 5;

    int red = (int)(val_of_cell(i - 1, j).red) * -1 
               + (int)(val_of_cell(i, j - 1).red) * -1 
               + (int)(val_of_cell(i + 1, j).red) * -1 
               + (int)(val_of_cell(i, j + 1).red) * -1 
               + (int)(val_of_cell(i, j).red) * 5; 

    int blue = (int)(val_of_cell(i - 1, j).blue) * -1 
               + (int)(val_of_cell(i, j - 1).blue) * -1 
               + (int)(val_of_cell(i + 1, j).blue) * -1 
               + (int)(val_of_cell(i, j + 1).blue) * -1 
               + (int)(val_of_cell(i, j).blue) * 5; 

    int alpha = (int)(val_of_cell(i - 1, j).alpha) * -1 
               + (int)(val_of_cell(i, j - 1).alpha) * -1 
               + (int)(val_of_cell(i + 1, j).alpha) * -1 
               + (int)(val_of_cell(i, j + 1).alpha) * -1 
               + (int)(val_of_cell(i, j).alpha) * 5; 
    
    B[n].green = green > 255 ? 255 : green;
    B[n].red = red > 255 ? 255 : red;
    B[n].blue = blue > 255 ? 255 : blue;
    B[n].alpha = alpha > 255 ? 255 : alpha;
  }
  else
  {
    B[n] = A[n];
  }
}

int main(int argc, char *argv[])
{
  if (argc == 1)
  {
    std::cout << "Input file name with bmp extension!" << std::endl;
    return;
  }
  std::string name = argv[1];

  auto img = readBMP(name);

  std::cout << "Img was read, data block size - " << img.data.size() << std::endl;

  reverse_colors f;
  thrust::transform(img.data.begin(), img.data.end(), img.data.begin(), f);

  cudaDeviceSynchronize();

  std::cout << "Colors were reversed" << std::endl;

  thrust::device_vector<Pixel> copyOfImg = img.data;
  
  Pixel* d_A = thrust::raw_pointer_cast(copyOfImg.data());
  Pixel* d_B = thrust::raw_pointer_cast(img.data.data());

  int threadsPerBlock = 1024;
  int blocksPerGrid = (img.data.size() + threadsPerBlock - 1) / threadsPerBlock;
  increase_contrast<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, img.info.height, img.info.width);
  cudaDeviceSynchronize();

  std::cout << cudaGetLastError() << std::endl;;

  std::cout << "Contrast was increased" << std::endl;

  writeBMP("./output.bmp", img);
}
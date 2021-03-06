// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if ( thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows )
    return;
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // int dim = blockDim.x; // assuming both dimension are equal
  int filter_length = filterWidth*filterWidth;
  extern __shared__ float sh_filter[];
  if (threadIdx.x == 0) {
    for (int i=0; i<filter_length; i++)
      sh_filter[i] = filter[i];

    // for (int r=0; r<dim; r++) {
    //   int image_r = min(max(blockIdx.y * blockDim.y + r, 0), static_cast<int>(numRows - 1));
    //   for (int c=0; c<dim; c++) {
    //     int image_c = min(max(blockIdx.x * blockDim.x + c, 0), static_cast<int>(numCols - 1));
    //     sh_filter[filter_length + r*dim + c] = static_cast<float>(inputChannel[image_r * numCols + image_c]);        
    //   }
    // }
  }
  // int image_r = min(max(thread_2D_pos.y, 0), static_cast<int>(numRows - 1));
  // int image_c = min(max(thread_2D_pos.x, 0), static_cast<int>(numCols - 1));
  // sh_filter[filter_length + threadIdx.y*dim + threadIdx.x] = static_cast<float>(inputChannel[image_r * numCols + image_c]);        
  __syncthreads();
  
  float result = 0.0f;
  for (int filter_r=-(filterWidth/2); filter_r<=filterWidth/2; filter_r++) {
    int image_r = min(max(thread_2D_pos.y + filter_r, 0), static_cast<int>(numRows - 1));
    // image_r -= blockIdx.y * blockDim.y;
    for (int filter_c=-(filterWidth/2); filter_c<=filterWidth/2; filter_c++) {
      int image_c = min(max(thread_2D_pos.x + filter_c, 0), static_cast<int>(numCols - 1));
      // image_c -= blockIdx.x * blockDim.x;
      // int loc = filter_length + image_r*dim + image_c;

      float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
      // float image_value = static_cast<float>(sh_filter[loc]);
      float filter_value = sh_filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];      
      
      result += image_value * filter_value;
    }    
  }

  outputChannel[thread_1D_pos] = result;
  // outputChannel[thread_1D_pos] = inputChannel[thread_1D_pos];
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if ( thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows )
    return;
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  uchar4 temp = inputImageRGBA[thread_1D_pos];
  redChannel[thread_1D_pos] = temp.x;
  greenChannel[thread_1D_pos] = temp.y;
  blueChannel[thread_1D_pos] = temp.z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if ( thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows )
    return;
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //Allocate memory for the filter on the GPU
  checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));
  //Copy the filter on the host (h_filter) to the memory you just allocated
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, 
                              sizeof(float) * filterWidth * filterWidth, 
                              cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const int dim = 32; // image is 512x288
  const dim3 blockSize(dim, dim, 1);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize((numCols + dim - 1)/dim, (numRows + dim - 1)/dim, 1);

  // float * h_filter = new float[filterWidth * filterWidth];
  //h_filter = (float *) malloc(sizeof(unsigned char) * filterWidth * filterWidth);

  // h_filter[0] = 0.0f; h_filter[1] = 0.2f; h_filter[2] = 0.0f;
  // h_filter[3] = 0.2f; h_filter[4] = 0.2f; h_filter[5] = 0.2f;
  // h_filter[6] = 0.0f; h_filter[7] = 0.2f; h_filter[8] = 0.0f;

  // h_filter[0] = 0.0f; h_filter[1] = 0.0f; h_filter[2] = 0.0f;
  // h_filter[3] = 0.0f; h_filter[4] = 0.0f; h_filter[5] = 0.0f;
  // h_filter[6] = 0.0f; h_filter[7] = 0.0f; h_filter[8] = 0.0f;
  
  // allocate memory for image, copy filter to gpu
  //allocateMemoryAndCopyToGPU(numRows, numCols, h_filter, filterWidth);
  // copy input image data from host to gpu
  // checkCudaErrors(cudaMalloc(&d_inputImageRGBA,  sizeof(uchar4) * numRows * numCols));
  // checkCudaErrors(cudaMemcpy(&d_inputImageRGBA, &h_inputImageRGBA, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice));
  // Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                             numRows,
                                             numCols,
                                             d_red,
                                             d_green,
                                             d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  // checkCudaErrors(cudaMalloc(&d_redBlurred,  sizeof(unsigned char) * numRows * numCols));
  // checkCudaErrors(cudaMalloc(&d_greenBlurred,  sizeof(unsigned char) * numRows * numCols));
  // checkCudaErrors(cudaMalloc(&d_blueBlurred,  sizeof(unsigned char) * numRows * numCols));
  gaussian_blur<<<gridSize, blockSize, sizeof(float)*(filterWidth*filterWidth/* + dim*dim*/)>>>(d_red,
                                         d_redBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  gaussian_blur<<<gridSize, blockSize, sizeof(float)*(filterWidth*filterWidth/* + dim*dim*/)>>>(d_green,
                                         d_greenBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  gaussian_blur<<<gridSize, blockSize, sizeof(float)*(filterWidth*filterWidth/* + dim*dim*/)>>>(d_blue,
                                         d_blueBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // checkCudaErrors(cudaFree(d_redBlurred));
  // checkCudaErrors(cudaFree(d_greenBlurred));
  // checkCudaErrors(cudaFree(d_blueBlurred));
  // checkCudaErrors(cudaFree(d_inputImageRGBA));
  // free(h_filter);
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}

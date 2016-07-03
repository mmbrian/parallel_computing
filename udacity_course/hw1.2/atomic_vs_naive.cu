#include <stdio.h>
#include "gputimer.h"
#include "scheme.h"

// #define NUM_THREADS 1000000
// #define ARRAY_SIZE  100

#define BLOCK_WIDTH 1000

void print_array(int *array, int size)
{
    printf("{ ");
    // for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
    for (int i = 0; i < 10; i++)  { printf("%d ", array[i]); }
    printf("}\n");
}

__global__ void increment_naive(int *g, int ARRAY_SIZE)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE; 
	g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g, int ARRAY_SIZE)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;  
	atomicAdd(& g[i], 1);
}

int main(int argc,char **argv)
{   
    testRun(1000000, 1000000, true);
    testRun(1000000, 1000000, false);
    testRun(1000000, 100, true);
    testRun(1000000, 100, false);
    testRun(10000000, 100, false);
    return 0;
}

void testRun(int NUM_THREADS, int ARRAY_SIZE, bool naive) {
    GpuTimer timer;
    
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // declare and allocate host memory
    // int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    //allocate the array
    // int h_array = new int[ARRAY_SIZE];
    int * h_array = new int[ARRAY_SIZE];
    memset((void *) h_array, 0, ARRAY_BYTES); 
 
    // declare, allocate, and zero out GPU memory
    int * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    // launch the kernel - comment out one of these
    timer.Start();
    
    // Instructions: This program is needed for the next quiz
    // uncomment increment_naive to measure speed and accuracy 
    // of non-atomic increments or uncomment increment_atomic to
    // measure speed and accuracy of  atomic icrements
    if (naive)
        increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array, ARRAY_SIZE);
    else 
        increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array, ARRAY_SIZE);
    timer.Stop();
    
    // copy back the array of sums from GPU and print
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed = %g ms\n", timer.Elapsed());
    
    delete [] h_array;

    // free GPU memory allocation and exit
    cudaFree(d_array);
}
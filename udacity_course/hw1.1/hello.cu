#include <stdio.h>

__global__ void hello(){
	printf("Hey there! from block %d, (Threads in block: %d, Blocks: %d)\n", 
		blockIdx.x, blockDim.x, gridDim.x);
}

int main(int argc, char ** argv) {
	
	// lunch kernel with 16 blocks and 1 thread each block
	hello<<<16, 1>>>();

	// force printf's to flush
	cudaDeviceSynchronize();

	printf("That's it\n");

	return 0;
}
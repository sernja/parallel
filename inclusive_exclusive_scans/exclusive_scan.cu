#include<stdio.h>
#include<math.h>

#define N 8

__global__ void exclusive_scan(int *d_in)
{
    //Phase 1 (Uptree)
    int s = 1;
    for(; s<=N-1; s<<=1)
    {
        int i = 2*s*(threadIdx.x+1)-1;
        if((i-s >= 0) && (i<N)) {
            //printf("s = %d, i= %d \n", s, i);
            int a = d_in[i];
            int b = d_in[i-s];
            __syncthreads();
            d_in[i] = a+b;
            //printf("Write in[%d] = %d\n", i, a+b);
        }
        __syncthreads();
    }

    //Phase 2 (Downtree)
    if(threadIdx.x == 0)
        d_in[N-1] = 0;
    
    for(s = s/2; s >= 1; s>>=1)
    {
        int i = 2*s*(threadIdx.x+1)-1;
        if((i-s >= 0) && (i<N)) {
            //printf("s = %d, i= %d \n", s, i);
            int r = d_in[i];
            int l = d_in[i-s];
            __syncthreads();
            d_in[i] = l+r;
            d_in[i-s] = r;
            __syncthreads();
            //printf("Write in[%d] = %d\n", i, a+b);
        }
        __syncthreads();
    }
}

int main()
{
	int h_in[N];
	int h_out[N];

	h_in[0] = 3;
    h_in[1] = 1;
    h_in[2] = 7;
    h_in[3] = 0;
    h_in[4] = 4;
    h_in[5] = 1;
    h_in[6] = 6;
    h_in[7] = 3;

	int *d_in;
	//int *d_out;

	cudaMalloc((void**) &d_in, N*sizeof(int));
	//cudaMalloc((void**) &d_out, N*sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);
	
	//Implementing kernel call
	exclusive_scan<<<1, 4>>>(d_in);

	cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++)
		printf("out[%d] =  %d\n", i, h_out[i]);

    cudaFree(d_in);

	return -1;
}

#include<stdio.h>
#include<math.h>
#define BLOCK_SIZE 1024

__global__ void multithreads_inverse_calculate(
        double* d_x_in, double* d_x_out, double entry_value, int d_n, int quantity, int entry_price, int leverage, int short_long
        )
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    //int tid = threadIdx.x;
    if(tid < d_n){
        //printf("d_x_in[%d]: %.2f\n", tid, d_x_in[tid]); //print d_x_in
        d_x_out[tid] = short_long*leverage*(quantity/entry_price-quantity/d_x_in[tid])/entry_value*100;
    }
} 

__global__ void multithreads_normal_calculate(
        double* d_x_in, double* d_xnormal_out, int d_n, int entry_price, int leverage, int short_long
        )
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    //int tid = threadIdx.x;
    if(tid < d_n){
        //printf("d_x_in[%d]: %.2f\n", tid, d_x_in[tid]); //print d_x_in
        d_xnormal_out[tid]
            = short_long*leverage*(d_x_in[tid]-entry_price)/entry_price*100;
    }
} 

int main(){

    int quantity, entry_price, exit_price, leverage, short_long;
    quantity = 1;
    entry_price = 1;
    exit_price = 1000;
    leverage = 1;
    short_long = -1;

    //ROE inverse calculate
    double entry_value, exit_value, profit, roe_inverse;
    entry_value = quantity/(double)entry_price;
    exit_value = quantity/(double)exit_price;
    profit = entry_value-exit_value;
    roe_inverse = (profit/entry_value)*100*leverage*short_long;

    //ROE normal calculate
    double roe_normal;
    roe_normal = short_long*leverage*((exit_price-entry_price)/(double)entry_price)*100;

    //find array x
    int num_arr;
    if(entry_price > exit_price){
        num_arr = (entry_price - exit_price);
    } else{
        num_arr = (exit_price - entry_price);
    }

    int n = num_arr*10+1;
    double x_in[n], xinverse_out[n], xnormal_out[n];

    for(int i = 0; i < n; i++){
        if(entry_price <= exit_price){
            x_in[i] = i*0.1 + entry_price; 
        } else{
            x_in[i] = i*0.1 + exit_price;
        }
    }

    //copy data from host to device
    double* d_x_in, *d_xinverse_out, *d_xnormal_out;
    cudaMalloc((void **) &d_x_in, n*sizeof(double));
    cudaMalloc((void **) &d_xinverse_out, n*sizeof(double));
    cudaMalloc((void **) &d_xnormal_out, n*sizeof(double));

    cudaMemcpy(d_x_in, &x_in, n*sizeof(double), cudaMemcpyHostToDevice);

    //time record start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    //Kernel launch
    /*multithreads_inverse_calculate<<<ceil(n/(double)BLOCK_SIZE), BLOCK_SIZE>>>(
            d_x_in, d_xinverse_out, entry_value, n, quantity, entry_price, leverage,short_long);*/

    //time record stop
    multithreads_normal_calculate<<<ceil(n/(double)BLOCK_SIZE), BLOCK_SIZE>>>(
            d_x_in,d_xnormal_out,n,entry_price,leverage,short_long);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float millisec = 0;
    cudaEventElapsedTime(&millisec, start, stop);

    //Copy data from device back to host. and free all data allocate on device
    cudaMemcpy(&xinverse_out, d_xinverse_out, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xnormal_out, d_xnormal_out, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x_in);
    cudaFree(d_xinverse_out);
    cudaFree(d_xnormal_out);

    //prit check output
    printf("\nROE inverse %.2lf%%\n", roe_inverse);
    printf("ROE normal %.2lf%%\n", roe_normal);
    printf("num arr: %d\n", num_arr);
    printf("n: %d\n", n);
    /*for(int i = 0; i < n; i++){
        //printf("%.2lf ", xinverse_out[i]);
    }
    for(int i=0; i<n; i++){
        printf("%.2lf" , xnormal_out[i]);
    }*/

    printf("Time: %.2f ms\n", millisec);
}

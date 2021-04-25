#include<stdio.h>
#include<math.h>
#define BLOCK_SIZE 10

__global__ void parallel_inverse_calculate(
        float* d_x_in, 
        float* d_x_out, 
        float entry_value, 
        int d_n, 
        int quantity, 
        int entry_price, 
        int leverage, 
        int short_long
        )
{
    //int tid = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    if(tid < d_n){
        printf("d_x_in[%d]: %.2f\n", tid, d_x_in[tid]); //print d_x_in
        d_x_out[tid] = short_long*leverage*(quantity/entry_price
                - quantity/d_x_in[tid])/entry_value*100;
    }
} 

int main(){

    int quantity, entry_price, exit_price, leverage, short_long;
    quantity = 1;
    entry_price = 1;
    exit_price = 10;
    leverage = 1;
    short_long = 1;

    //ROE inverse calculate
    float entry_value, exit_value, profit, roe_inverse;
    entry_value = (float)quantity/(float)entry_price;
    exit_value = (float)quantity/(float)exit_price;
    profit = entry_value-exit_value;
    roe_inverse = (profit/entry_value)*100*leverage*short_long;

    //ROE normal calculate
    float roe_normal;
    roe_normal = short_long*leverage*((exit_price-entry_price)/(float)entry_price)*100;

    //find array x
    int num_arr;
    if(entry_price > exit_price){
        num_arr = (entry_price - exit_price);
    } else{
        num_arr = (exit_price - entry_price);
    }

    int n = num_arr*10+1;
    float x_in[n];
    float x_out[n];

    printf("x_in: \n");
    for(int i = 0; i < n; i++){
        if(entry_price <= exit_price){
            x_in[i] = i*0.1 + entry_price; 
        } else{
            x_in[i] = i*0.1 + exit_price;
        }
        printf("%.2f ", x_in[i]);
    }
    printf("\n\n");

    //copy data from host to device
    float* d_x_in, *d_x_out;
    cudaMalloc((void **) &d_x_in, n*sizeof(float));
    cudaMalloc((void **) &d_x_out, n*sizeof(float));

    cudaMemcpy(d_x_in, &x_in, n*sizeof(float), cudaMemcpyHostToDevice);

    //Kernel launch
    //parallel_inverse_calculate<<<
      //  ceil(num_arr/(double) BLOCK_SIZE),BLOCK_SIZE
        //>>>(d_x_in, d_x_out, entry_value, n, quantity, entry_price, leverage, 
        //short_long);
    parallel_inverse_calculate<<<1,100>>>
        (d_x_in, d_x_out, entry_value, n, quantity, entry_price, leverage, 
        short_long);
    cudaDeviceSynchronize();

    //Copy data from device back to host. and free all data allocate on device
    cudaMemcpy(&x_out, d_x_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x_in);
    cudaFree(d_x_out);

    //prit check output
    printf("\nROE inverse %.2f%%\n", roe_inverse);
    printf("ROE normal %.2f%%\n", roe_normal);
    printf("num arr: %d\n", num_arr);
    printf("n: %d\n", n);
    for(int i = 0; i < n; i++){
        printf("%.2f ", x_out[i]);
    }
    printf("\nlen x_in: %ld\n", sizeof(x_in));
}

#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUBLAS(Expr) { \
    int err = (Expr); \
    if (err != 0) { \
        printf("cuBLAS error %d at line %d\n", err, __LINE__); \
        exit(1); \
    } \
}

#define CHECK_CUDA(Expr) { \
    cudaError_t err = (Expr); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

void gemm(cublasHandle_t handle,
          int m,
          int n,
          int k,
          const void *alpha,
          const void *beta,
          cudaDataType_t input_type,
          const void *A,
          const void *B,
          cudaDataType_t output_type,
          void *C,
#if __CUDACC_VER_MAJOR__ >= 11
          cublasComputeType_t compute_type,
#else
          cudaDataType_t compute_type,
#endif
          cublasGemmAlgo_t algo) {
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        alpha, B, input_type, n, A, input_type, k,
        beta, C, output_type, n, compute_type, algo));
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;

    float alpha = 1;
    float beta = 0;

    cudaDataType_t input_type = CUDA_R_32F;
    cudaDataType_t output_type = CUDA_R_32F;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    int iter = 10;

    void *A, *B, *C;
    // Fix: Add proper error checking for CUDA memory allocations
    CHECK_CUDA(cudaMalloc(&A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C, m * n * sizeof(float)));

    cublasHandle_t handle;
    // Fix: Add error checking for cuBLAS handle creation
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
         output_type, C, compute_type, algo);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iter; ++i) {
        gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
             output_type, C, compute_type, algo);
    }
    CHECK_CUDA(cudaEventRecord(stop));

    float time_ms = 0.f;
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));

    long ops = (long)m * n * k * 2;
    double gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    printf("%f Gops\n", gops);

    // Fix: Proper cleanup of resources
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    
    return 0;
}



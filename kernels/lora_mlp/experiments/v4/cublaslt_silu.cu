#include <torch/extension.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>

static void check_cublaslt(cublasStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLt error ", status, ": ", msg);
}

torch::Tensor cublaslt_matmul_with_epilogue(
    torch::Tensor A,    // [M, K]
    torch::Tensor B,    // [K, N]
    float alpha_val,
    int epilogue_type   // 1=none, 2=relu, 32=gelu, 2048=swish/silu
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.is_contiguous() && B.is_contiguous());
    TORCH_CHECK(A.size(1) == B.size(0), "Shape mismatch");

    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto D = torch::empty({M, N}, A.options());

    cudaDataType_t dt;
    if (A.scalar_type() == torch::kBFloat16) dt = CUDA_R_16BF;
    else if (A.scalar_type() == torch::kFloat16) dt = CUDA_R_16F;
    else dt = CUDA_R_32F;

    cublasLtHandle_t handle;
    check_cublaslt(cublasLtCreate(&handle), "create handle");

    cublasLtMatmulDesc_t desc;
    check_cublaslt(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create desc");

    cublasOperation_t op_n = CUBLAS_OP_N;
    check_cublaslt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)), "set transa");
    check_cublaslt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)), "set transb");

    if (epilogue_type != 1) {
        cublasLtEpilogue_t epi = static_cast<cublasLtEpilogue_t>(epilogue_type);
        check_cublaslt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)), "set epilogue");
    }

    // Row-major A[M,K] @ B[K,N] = D[M,N]
    // Col-major: D^T[N,M] = B_cm[N,K] @ A_cm[K,M]
    cublasLtMatrixLayout_t la, lb, lc, ld;
    check_cublaslt(cublasLtMatrixLayoutCreate(&la, dt, N, K, N), "layout A");
    check_cublaslt(cublasLtMatrixLayoutCreate(&lb, dt, K, M, K), "layout B");
    check_cublaslt(cublasLtMatrixLayoutCreate(&lc, dt, N, M, N), "layout C");
    check_cublaslt(cublasLtMatrixLayoutCreate(&ld, dt, N, M, N), "layout D");

    cublasLtMatmulPreference_t pref;
    check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "create pref");
    size_t ws_size = 32 * 1024 * 1024;
    check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)), "set ws");

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld, pref, 1, &heuristic, &returned);
    check_cublaslt(status, "get heuristic");
    TORCH_CHECK(returned > 0, "No algorithm found for this cublasLt config (epilogue=", epilogue_type, ")");

    auto workspace = torch::empty({(int64_t)ws_size}, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
    float alpha = alpha_val, beta = 0.0f;

    check_cublaslt(cublasLtMatmul(
        handle, desc, &alpha,
        B.data_ptr(), la,
        A.data_ptr(), lb,
        &beta,
        D.data_ptr(), lc,
        D.data_ptr(), ld,
        &heuristic.algo,
        workspace.data_ptr(), ws_size,
        at::cuda::getCurrentCUDAStream()
    ), "matmul");

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);

    return D;
}

torch::Tensor cublaslt_matmul_silu(torch::Tensor A, torch::Tensor B, float alpha) {
    return cublaslt_matmul_with_epilogue(A, B, alpha, 2048);
}

torch::Tensor cublaslt_matmul_plain(torch::Tensor A, torch::Tensor B, float alpha) {
    return cublaslt_matmul_with_epilogue(A, B, alpha, 1);
}

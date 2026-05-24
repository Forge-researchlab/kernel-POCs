#include <torch/extension.h>

torch::Tensor cublaslt_matmul_silu(torch::Tensor A, torch::Tensor B, float alpha);
torch::Tensor cublaslt_matmul_plain(torch::Tensor A, torch::Tensor B, float alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublaslt_matmul_silu", &cublaslt_matmul_silu, "cublasLt matmul with SiLU epilogue");
    m.def("cublaslt_matmul_plain", &cublaslt_matmul_plain, "cublasLt plain matmul");
}

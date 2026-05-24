"""
Raw ctypes wrapper for cublasLt with fused epilogue support (SiLU/ReLU/GELU).

This calls NVIDIA's cublasLtMatmul directly via ctypes, bypassing PyTorch's
limited interface. The key feature is the CUBLASLT_EPILOGUE_SWISH epilogue
which fuses SiLU into the GEMM at full cuBLAS speed — zero extra HBM traffic.

API: cublaslt_matmul_epilogue(A, B, epilogue="swish") -> D = epilogue(A @ B)

Supported epilogues:
  - "none":  D = alpha * A @ B + beta * C
  - "relu":  D = ReLU(alpha * A @ B + beta * C)
  - "gelu":  D = GELU(alpha * A @ B + beta * C)
  - "swish": D = SiLU(alpha * A @ B + beta * C)  <-- this is the prize
"""

import ctypes
import torch

# Load cublasLt shared library
_cublaslt = ctypes.cdll.LoadLibrary("libcublasLt.so.12")

# ─── cublasLt constants ───

# cublasComputeType_t
CUBLAS_COMPUTE_32F = 68

# cudaDataType_t
CUDA_R_16BF = 14
CUDA_R_16F = 2
CUDA_R_32F = 0

# cublasLtOrder_t
CUBLASLT_ORDER_ROW = 1
CUBLASLT_ORDER_COL = 0

# cublasOperation_t
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

# cublasLtMatmulDescAttributes_t
CUBLASLT_MATMUL_DESC_TRANSA = 0
CUBLASLT_MATMUL_DESC_TRANSB = 1
CUBLASLT_MATMUL_DESC_EPILOGUE = 2

# cublasLtEpilogue_t
CUBLASLT_EPILOGUE_DEFAULT = 1
CUBLASLT_EPILOGUE_RELU = 2
CUBLASLT_EPILOGUE_GELU = 32
CUBLASLT_EPILOGUE_SWISH = 2048

# cublasLtMatmulPreferenceAttributes_t
CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1

EPILOGUE_MAP = {
    "none": CUBLASLT_EPILOGUE_DEFAULT,
    "relu": CUBLASLT_EPILOGUE_RELU,
    "gelu": CUBLASLT_EPILOGUE_GELU,
    "swish": CUBLASLT_EPILOGUE_SWISH,
    "silu": CUBLASLT_EPILOGUE_SWISH,
}

DTYPE_MAP = {
    torch.float32: CUDA_R_32F,
    torch.float16: CUDA_R_16F,
    torch.bfloat16: CUDA_R_16BF,
}

# ─── ctypes type aliases ───

_c_void_p = ctypes.c_void_p
_c_int = ctypes.c_int
_c_int32 = ctypes.c_int32
_c_uint32 = ctypes.c_uint32
_c_uint64 = ctypes.c_uint64
_c_size_t = ctypes.c_size_t
_c_float = ctypes.c_float


def _check(status, msg=""):
    if status != 0:
        raise RuntimeError(f"cublasLt error {status}: {msg}")


# ─── Handle management ───

_handle = _c_void_p()
_check(_cublaslt.cublasLtCreate(ctypes.byref(_handle)), "cublasLtCreate")


def _get_cuda_stream():
    return ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(torch.cuda.current_device()))


# ─── Core function ───

def cublaslt_matmul_epilogue(
    A: torch.Tensor,
    B: torch.Tensor,
    epilogue: str = "none",
    alpha: float = 1.0,
    beta: float = 0.0,
    C: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute D = epilogue(alpha * A @ B + beta * C) using cublasLtMatmul.

    A: [M, K] (row-major)
    B: [K, N] (row-major) — note: pass W.t() if W is [N, K]
    C: [M, N] optional bias/accumulator
    epilogue: "none", "relu", "gelu", "swish"/"silu"

    Returns D: [M, N]
    """
    assert A.is_cuda and B.is_cuda and A.is_contiguous() and B.is_contiguous()
    assert A.dtype == B.dtype
    assert A.shape[1] == B.shape[0], f"Shape mismatch: A={A.shape}, B={B.shape}"

    M, K = A.shape
    N = B.shape[1]
    dtype = A.dtype

    D = torch.empty(M, N, dtype=dtype, device=A.device)
    if C is None:
        C = D

    cuda_dtype = DTYPE_MAP[dtype]
    epilogue_val = EPILOGUE_MAP[epilogue]

    # ── Create matmul descriptor ──
    matmul_desc = _c_void_p()
    _check(_cublaslt.cublasLtMatmulDescCreate(
        ctypes.byref(matmul_desc),
        _c_uint32(CUBLAS_COMPUTE_32F),
        _c_uint32(CUDA_R_32F),
    ), "MatmulDescCreate")

    # cublasLt is column-major by default. For row-major A[M,K] @ B[K,N]:
    # We tell cublasLt: C = B^T @ A^T (swapped + transposed), then the result is column-major
    # which matches row-major C[M,N].
    transa = _c_int32(CUBLAS_OP_N)
    transb = _c_int32(CUBLAS_OP_N)
    _check(_cublaslt.cublasLtMatmulDescSetAttribute(
        matmul_desc,
        _c_uint32(CUBLASLT_MATMUL_DESC_TRANSA),
        ctypes.byref(transa), ctypes.sizeof(transa),
    ), "SetAttribute TRANSA")
    _check(_cublaslt.cublasLtMatmulDescSetAttribute(
        matmul_desc,
        _c_uint32(CUBLASLT_MATMUL_DESC_TRANSB),
        ctypes.byref(transb), ctypes.sizeof(transb),
    ), "SetAttribute TRANSB")

    # Set epilogue
    epi = _c_uint32(epilogue_val)
    _check(_cublaslt.cublasLtMatmulDescSetAttribute(
        matmul_desc,
        _c_uint32(CUBLASLT_MATMUL_DESC_EPILOGUE),
        ctypes.byref(epi), ctypes.sizeof(epi),
    ), "SetAttribute EPILOGUE")

    # ── Create matrix layouts ──
    # cublasLt uses column-major by default.
    # For row-major: swap M/N and use ld = number of columns.
    # D[M,N] row-major = D[N,M] column-major with ld=N
    # But simpler: compute D_col = B_col @ A_col where:
    #   B is [K,N] row-major = [N,K] col-major
    #   A is [M,K] row-major = [K,M] col-major
    #   D = [M,N] row-major = [N,M] col-major
    # So cublasLt sees: D[N,M] = B[N,K] @ A[K,M] → m=N, n=M, k=K

    m_lt, n_lt, k_lt = N, M, K

    def make_layout(rows, cols, ld):
        layout = _c_void_p()
        _check(_cublaslt.cublasLtMatrixLayoutCreate(
            ctypes.byref(layout),
            _c_uint32(cuda_dtype),
            _c_uint64(rows),
            _c_uint64(cols),
            _c_uint64(ld),
        ), "MatrixLayoutCreate")
        return layout

    # B is the "A" matrix for cublasLt (col-major: [N, K] with ld=N)
    layout_a = make_layout(m_lt, k_lt, B.stride(0))
    # A is the "B" matrix for cublasLt (col-major: [K, M] with ld=K)
    layout_b = make_layout(k_lt, n_lt, A.stride(0))
    # C and D (col-major: [N, M] with ld=N)
    layout_c = make_layout(m_lt, n_lt, C.stride(0) if C.data_ptr() != D.data_ptr() else N)
    layout_d = make_layout(m_lt, n_lt, D.stride(0))

    # ── Algorithm selection via heuristic ──
    preference = _c_void_p()
    _check(_cublaslt.cublasLtMatmulPreferenceCreate(ctypes.byref(preference)), "PrefCreate")

    workspace_size = _c_uint64(32 * 1024 * 1024)  # 32 MB workspace
    _check(_cublaslt.cublasLtMatmulPreferenceSetAttribute(
        preference,
        _c_uint32(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES),
        ctypes.byref(workspace_size), ctypes.sizeof(workspace_size),
    ), "PrefSetWorkspace")

    # heuristicResult struct is 1120 bytes (opaque, we just need the first one)
    heuristic_buf = (ctypes.c_byte * 1120)()
    returned_results = _c_int(0)

    status = _cublaslt.cublasLtMatmulAlgoGetHeuristic(
        _handle,
        matmul_desc,
        layout_a, layout_b, layout_c, layout_d,
        preference,
        _c_int(1),
        ctypes.byref(heuristic_buf),
        ctypes.byref(returned_results),
    )
    _check(status, f"AlgoGetHeuristic (returned={returned_results.value})")

    if returned_results.value == 0:
        raise RuntimeError("cublasLt: no algorithm found for this config")

    # The algo is at offset 0 of the heuristic result (first field)
    algo_ptr = ctypes.cast(ctypes.byref(heuristic_buf), _c_void_p)

    # ── Workspace ──
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=A.device)

    # ── Run matmul ──
    alpha_f = _c_float(alpha)
    beta_f = _c_float(beta)
    stream = _get_cuda_stream()

    status = _cublaslt.cublasLtMatmul(
        _handle,
        matmul_desc,
        ctypes.byref(alpha_f),
        _c_void_p(B.data_ptr()),  # "A" for cublasLt (swapped for row-major)
        layout_a,
        _c_void_p(A.data_ptr()),  # "B" for cublasLt
        layout_b,
        ctypes.byref(beta_f),
        _c_void_p(C.data_ptr()),
        layout_c,
        _c_void_p(D.data_ptr()),
        layout_d,
        algo_ptr,
        _c_void_p(workspace.data_ptr()),
        _c_size_t(workspace.numel()),
        stream,
    )
    _check(status, "cublasLtMatmul")

    # ── Cleanup ──
    _cublaslt.cublasLtMatmulPreferenceDestroy(preference)
    _cublaslt.cublasLtMatrixLayoutDestroy(layout_a)
    _cublaslt.cublasLtMatrixLayoutDestroy(layout_b)
    _cublaslt.cublasLtMatrixLayoutDestroy(layout_c)
    _cublaslt.cublasLtMatrixLayoutDestroy(layout_d)
    _cublaslt.cublasLtMatmulDescDestroy(matmul_desc)

    return D

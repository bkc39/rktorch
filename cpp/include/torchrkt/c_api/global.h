#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* tr_version(void);

/* SHA-256 of the c_api headers this shim was built from. */
const char* tr_abi_stamp(void);

/* Thread-local, as is tr_last_error_kind. */
const char* tr_last_error(void);

/* 0 generic, 1 out-of-memory. */
int tr_last_error_kind(void);

int tr_manual_seed(uint64_t seed);

/* Process-wide flags of at::globalContext(), shared by every thread: a write
 * is seen by every op dispatched after it on any thread, and nothing orders
 * it against an op already running. The cuDNN flags are stored on a build
 * without CUDA too. Each getter writes *out (0 or 1) and returns 0, or
 * returns 1 on a NULL out. */
int tr_set_deterministic_algorithms(int enabled, int warn_only);
int tr_get_deterministic_algorithms(int* out);
int tr_get_deterministic_algorithms_warn_only(int* out);
int tr_set_cudnn_deterministic(int enabled);
int tr_get_cudnn_deterministic(int* out);
int tr_set_cudnn_benchmark(int enabled);
int tr_get_cudnn_benchmark(int* out);

/* TF32 and float32 matmul precision, flags of the same context. A precision
 * code is 0 highest, 1 high, 2 medium, as at::Float32MatmulPrecision; the
 * setter returns 1 on any other code and writes nothing. The precision getter
 * returns 1 and leaves *out unwritten where libtorch refuses to answer:
 * after tr_set_cuda_matmul_allow_tf32 has contradicted a high or medium
 * precision. */
int tr_set_float32_matmul_precision(int precision);
int tr_get_float32_matmul_precision(int* out);
int tr_set_cuda_matmul_allow_tf32(int enabled);
int tr_get_cuda_matmul_allow_tf32(int* out);
int tr_set_cudnn_allow_tf32(int enabled);
int tr_get_cudnn_allow_tf32(int* out);

#ifdef __cplusplus
}
#endif

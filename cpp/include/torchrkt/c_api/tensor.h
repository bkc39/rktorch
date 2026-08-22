#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle owning a single torch::Tensor. Create one with tr_randn (see
 * random.h); release it with tr_tensor_free. All accessors below take the
 * same size-then-fill / integer-status contract as the rest of the C API:
 *   0: success
 *   1: error  — call tr_last_error()
 *   2: caller buffer too small — the required size is written to the *out_*
 *      count and nothing is copied. */
typedef struct tr_tensor tr_tensor;

/* Element types reachable from Racket. Values are part of the C ABI, and so
 * is the enum's width: the Racket FFI marshals it as a C int, so a narrower
 * base type would be an ABI break (and C can't specify one before C23). */
/* NOLINTNEXTLINE(performance-enum-size) */
typedef enum tr_dtype {
  TR_DTYPE_FLOAT32 = 0,
  TR_DTYPE_FLOAT64 = 1,
  TR_DTYPE_INT64 = 2,
  TR_DTYPE_BOOL = 3
} tr_dtype;

/* Free a handle returned by tr_randn. Safe on NULL. If the underlying
 * storage release fails (e.g. a CUDA context error), the failure cannot be
 * intercepted at this layer — it terminates inside libtorch's noexcept
 * release path (see tests/torchrkt/finalizer_death_test.cpp) — so direct C
 * callers get no error report; the Racket binding layer adds its own
 * finalizer-side guard for the failure classes it can observe. */
void tr_tensor_free(tr_tensor* t);

/* Total number of elements. */
int tr_tensor_numel(const tr_tensor* t, int64_t* out);

/* Total bytes of the tensor's element data: numel x element size — the
 * view's extent, not the underlying (possibly shared) storage's. */
int tr_tensor_nbytes(const tr_tensor* t, int64_t* out);

/* Number of dimensions. */
int tr_tensor_ndim(const tr_tensor* t, int64_t* out);

/* Copy the dimension sizes into out_dims (capacity in elements). *out_ndim
 * always receives the true ndim; rc=2 if capacity < ndim. */
int tr_tensor_shape(const tr_tensor* t, int64_t capacity, int64_t* out_dims,
                    int64_t* out_ndim);

/* Copy the tensor's values as row-major float32 into out (capacity in
 * elements). The tensor is moved to CPU + float32 + contiguous first.
 * *out_numel always receives the true element count; rc=2 if capacity <
 * numel. */
int tr_tensor_copy_data(const tr_tensor* t, uint64_t capacity, float* out,
                        uint64_t* out_numel);

/* Extract the value of a single-element tensor as a double (like
 * torch.Tensor.item). Errors if the tensor has more than one element. */
int tr_tensor_item(const tr_tensor* t, double* out);

/* Copy converted to the given element type (like torch.Tensor.to). Returns a
 * new handle, NULL on error. */
tr_tensor* tr_tensor_to_dtype(const tr_tensor* t, tr_dtype dtype);

/* Report a tensor's dtype. 0: success with *out set; 1 (see
 * tr_last_error) for NULL args or a dtype outside the tr_dtype enum. */
int tr_tensor_dtype(const tr_tensor* t, tr_dtype* out);

/* tr_tensor_copy_data's int64 sibling: copies via a CPU/int64/contiguous
 * conversion, so integer tensors round-trip exactly instead of through
 * float32. Same size-then-fill contract. */
int tr_tensor_copy_data_i64(const tr_tensor* t, uint64_t capacity, int64_t* out,
                            uint64_t* out_numel);

/* The float64 sibling: double-precision values marshal without float32
 * truncation. Same size-then-fill contract. */
int tr_tensor_copy_data_f64(const tr_tensor* t, uint64_t capacity, double* out,
                            uint64_t* out_numel);

/* Render the tensor via ATen's ostream operator into out_buffer (capacity in
 * bytes, no NUL terminator written). *out_len always receives the byte length;
 * rc=2 if buffer_capacity < len. */
int tr_tensor_print(const tr_tensor* t, uint64_t buffer_capacity,
                    char* out_buffer, uint64_t* out_len);

#ifdef __cplusplus
}
#endif

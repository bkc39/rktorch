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

/* Free a handle returned by tr_randn. Safe on NULL. */
void tr_tensor_free(tr_tensor* t);

/* Total number of elements. */
int tr_tensor_numel(const tr_tensor* t, int64_t* out);

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

/* Render the tensor via ATen's ostream operator into out_buffer (capacity in
 * bytes, no NUL terminator written). *out_len always receives the byte length;
 * rc=2 if buffer_capacity < len. */
int tr_tensor_print(const tr_tensor* t, uint64_t buffer_capacity,
                    char* out_buffer, uint64_t* out_len);

#ifdef __cplusplus
}
#endif

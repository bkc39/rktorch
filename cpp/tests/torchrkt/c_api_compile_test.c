/* Compile-only: prove the public headers are valid C and every entry point has
 * C linkage. Nothing here runs; it is linked into the gtest binary as an object
 * so a C++-only leak in the headers fails the build. */
#include <stdbool.h>
#include <stdint.h>

#include "torchrkt/c_api.h"

void torchrkt_c_api_compile_check(void);

void torchrkt_c_api_compile_check(void) {
  const char* (*version)(void) = tr_version;
  const char* (*last_error)(void) = tr_last_error;
  int (*manual_seed)(uint64_t) = tr_manual_seed;
  tr_tensor* (*randn)(const int64_t*, int64_t) = tr_randn;
  void (*tensor_free)(tr_tensor*) = tr_tensor_free;
  int (*numel)(const tr_tensor*, int64_t*) = tr_tensor_numel;
  int (*nbytes)(const tr_tensor*, int64_t*) = tr_tensor_nbytes;
  int (*copy_data)(const tr_tensor*, uint64_t, float*, uint64_t*) =
      tr_tensor_copy_data;
  /* One representative per new op family (creation, shape, elementwise,
   * reduce, linalg, marshalling, rng). */
  tr_tensor* (*zeros)(const int64_t*, int64_t) = tr_zeros;
  tr_tensor* (*from_data)(const float*, uint64_t, const int64_t*, int64_t) =
      tr_from_data;
  tr_tensor* (*from_data_on)(const float*, uint64_t, const int64_t*, int64_t,
                             tr_device_type, int64_t) = tr_from_data_on;
  tr_tensor* (*cat)(const tr_tensor* const*, int64_t, int64_t) = tr_cat;
  tr_tensor* (*add)(const tr_tensor*, const tr_tensor*) = tr_add;
  tr_tensor* (*add_scalar)(const tr_tensor*, double) = tr_add_scalar;
  tr_tensor* (*gelu)(const tr_tensor*) = tr_gelu;
  tr_tensor* (*softmax)(const tr_tensor*, int64_t) = tr_softmax;
  tr_tensor* (*matmul)(const tr_tensor*, const tr_tensor*) = tr_matmul;
  int (*item)(const tr_tensor*, double*) = tr_tensor_item;
  tr_tensor* (*to_dtype)(const tr_tensor*, tr_dtype) = tr_tensor_to_dtype;
  tr_tensor* (*rand_fn)(const int64_t*, int64_t) = tr_rand;
  int (*uniform_fn)(tr_tensor*, double, double) = tr_tensor_uniform_;
  int (*requires_grad_set)(tr_tensor*, int) = tr_tensor_requires_grad_;
  int (*has_grad)(const tr_tensor*, int*) = tr_tensor_has_grad;
  int (*backward)(tr_tensor*) = tr_tensor_backward;
  tr_tensor* (*grad)(const tr_tensor*) = tr_tensor_grad;
  int (*set_grad_enabled)(int) = tr_set_grad_enabled;
  int (*sub_inplace)(tr_tensor*, const tr_tensor*, double) = tr_tensor_sub_;
  /* The generated surface: one pin per generated *signature shape* (the
   * include only proves the headers parse as C; the pointer assignment
   * proves the symbol links with C linkage). matmul covers tensor-tensor,
   * reshape covers IntArrayRef, cat covers TensorList. */
  tr_tensor* (*gen_matmul)(const tr_tensor*, const tr_tensor*) = tr_gen_matmul;
  tr_tensor* (*gen_reshape)(const tr_tensor*, const int64_t*, int64_t) =
      tr_gen_reshape;
  tr_tensor* (*gen_cat)(const tr_tensor* const*, int64_t, int64_t) = tr_gen_cat;
  /* tranche-2 added four shapes: an int-status in-place op (mutable
   * receiver), an optional-int-array with a `_has` presence flag, an
   * optional-int64 (`int64_t n, bool n_has`), and an optional-tensor whose
   * NULL encodes c10::nullopt (conv2d bias, loss weight). */
  int (*gen_mul_)(tr_tensor*, const tr_tensor*) = tr_gen_mul__tensor;
  /* the in-place family has four distinct arg shapes; pin each. */
  int (*gen_add_)(tr_tensor*, const tr_tensor*, double) = tr_gen_add__tensor;
  int (*gen_lerp_)(tr_tensor*, const tr_tensor*, const tr_tensor*) =
      tr_gen_lerp__tensor;
  int (*gen_addcmul_)(tr_tensor*, const tr_tensor*, const tr_tensor*, double) =
      tr_gen_addcmul_;
  /* copy_: in-place with a trailing bool (a new in-place arg shape). */
  int (*gen_copy_)(tr_tensor*, const tr_tensor*, bool) = tr_gen_copy_;
  tr_tensor* (*gen_sum_dim)(const tr_tensor*, const int64_t*, int64_t, bool,
                            bool, int32_t) = tr_gen_sum_dim_intlist;
  tr_tensor* (*gen_conv2d)(const tr_tensor*, const tr_tensor*, const tr_tensor*,
                           const int64_t*, int64_t, const int64_t*, int64_t,
                           const int64_t*, int64_t, int64_t) = tr_gen_conv2d;
  tr_tensor* (*gen_avg_pool2d)(const tr_tensor*, const int64_t*, int64_t,
                               const int64_t*, int64_t, const int64_t*, int64_t,
                               bool, bool, int64_t, bool) = tr_gen_avg_pool2d;
  /* one representative per new generated family header (compare.h, loss.h)
   * and per distinct signature shape within them. */
  tr_tensor* (*gen_eq_tensor)(const tr_tensor*, const tr_tensor*) =
      tr_gen_eq_tensor;
  tr_tensor* (*gen_eq_scalar)(const tr_tensor*, double) = tr_gen_eq_scalar;
  tr_tensor* (*gen_nll_loss)(const tr_tensor*, const tr_tensor*,
                             const tr_tensor*, int64_t, int64_t) =
      tr_gen_nll_loss;
  tr_tensor* (*gen_cross_entropy_loss)(const tr_tensor*, const tr_tensor*,
                                       const tr_tensor*, int64_t, int64_t,
                                       double) = tr_gen_cross_entropy_loss;
  tr_tensor* (*gen_max_pool2d)(const tr_tensor*, const int64_t*, int64_t,
                               const int64_t*, int64_t, const int64_t*, int64_t,
                               const int64_t*, int64_t, bool) =
      tr_gen_max_pool2d;
  /* tranche-3 signature shapes: tensor-tensor with trailing int64+bools
   * (embedding), IntArrayRef with two optional-tensors (layer_norm),
   * tensor-tensor-scalar (masked_fill), and tensor+int64 (tril/triu; one
   * pin covers both). */
  tr_tensor* (*gen_embedding)(const tr_tensor*, const tr_tensor*, int64_t, bool,
                              bool) = tr_gen_embedding;
  tr_tensor* (*gen_layer_norm)(const tr_tensor*, const int64_t*, int64_t,
                               const tr_tensor*, const tr_tensor*, double,
                               bool) = tr_gen_layer_norm;
  tr_tensor* (*gen_masked_fill)(const tr_tensor*, const tr_tensor*, double) =
      tr_gen_masked_fill_scalar;
  tr_tensor* (*gen_tril)(const tr_tensor*, int64_t) = tr_gen_tril;
  tr_tensor* (*gen_adaptive_avg_pool2d)(const tr_tensor*, const int64_t*,
                                        int64_t) = tr_gen_adaptive_avg_pool2d;
  /* narrow: tensor + three plain int64 scalars (a new shape). */
  tr_tensor* (*gen_narrow)(const tr_tensor*, int64_t, int64_t, int64_t) =
      tr_gen_narrow;
  /* dropout: tensor + double + bool (a new shape). */
  tr_tensor* (*gen_dropout)(const tr_tensor*, double, bool) = tr_gen_dropout;
  /* device family (device.h): the cuda queries, the default-device pair, and
   * the per-tensor to/from device functions. */
  int (*cuda_available)(void) = tr_cuda_is_available;
  int (*cuda_count)(void) = tr_cuda_device_count;
  int (*set_default_device)(tr_device_type, int64_t) = tr_set_default_device;
  int (*get_default_device)(tr_device_type*, int64_t*) = tr_get_default_device;
  tr_tensor* (*to_device)(const tr_tensor*, tr_device_type, int64_t) =
      tr_tensor_to_device;
  int (*tensor_device)(const tr_tensor*, tr_device_type*, int64_t*) =
      tr_tensor_device;

  (void)version;
  (void)last_error;
  (void)manual_seed;
  (void)randn;
  (void)tensor_free;
  (void)numel;
  (void)nbytes;
  (void)copy_data;
  (void)zeros;
  (void)from_data;
  (void)cat;
  (void)add;
  (void)add_scalar;
  (void)softmax;
  (void)matmul;
  (void)item;
  (void)to_dtype;
  (void)rand_fn;
  (void)uniform_fn;
  (void)requires_grad_set;
  (void)has_grad;
  (void)backward;
  (void)grad;
  (void)set_grad_enabled;
  (void)sub_inplace;
  (void)gen_matmul;
  (void)gen_reshape;
  (void)gen_cat;
  (void)gen_mul_;
  (void)gen_add_;
  (void)gen_lerp_;
  (void)gen_addcmul_;
  (void)gen_copy_;
  (void)gen_sum_dim;
  (void)gen_conv2d;
  (void)gen_avg_pool2d;
  (void)gen_adaptive_avg_pool2d;
  (void)gen_eq_tensor;
  (void)gen_eq_scalar;
  (void)gen_nll_loss;
  (void)gen_cross_entropy_loss;
  (void)gen_max_pool2d;
  (void)gen_narrow;
  (void)gen_dropout;
  (void)cuda_available;
  (void)cuda_count;
  (void)set_default_device;
  (void)get_default_device;
  (void)to_device;
  (void)tensor_device;
}

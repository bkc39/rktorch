/* Compile-only: prove the public headers are valid C and every entry point has
 * C linkage. Nothing here runs; it is linked into the gtest binary as an object
 * so a C++-only leak in the headers fails the build. */
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
  int (*copy_data)(const tr_tensor*, uint64_t, float*, uint64_t*) =
      tr_tensor_copy_data;
  /* One representative per new op family (creation, shape, elementwise,
   * reduce, linalg, marshalling, rng). */
  tr_tensor* (*zeros)(const int64_t*, int64_t) = tr_zeros;
  tr_tensor* (*from_data)(const float*, uint64_t, const int64_t*, int64_t) =
      tr_from_data;
  tr_tensor* (*cat)(const tr_tensor* const*, int64_t, int64_t) = tr_cat;
  tr_tensor* (*add)(const tr_tensor*, const tr_tensor*) = tr_add;
  tr_tensor* (*add_scalar)(const tr_tensor*, double) = tr_add_scalar;
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
  /* The generated surface: the golden linalg four are permanent allowlist
   * entries, so their linkage is pinned here; other generated ops prove C
   * compatibility via the c_api/generated.h include alone. */
  tr_tensor* (*gen_matmul)(const tr_tensor*, const tr_tensor*) = tr_gen_matmul;

  (void)version;
  (void)last_error;
  (void)manual_seed;
  (void)randn;
  (void)tensor_free;
  (void)numel;
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
}

/* Compile-only: pins C linkage for every public entry point. */
#include <stdbool.h>
#include <stdint.h>

#include "torchrkt/c_api.h"

void torchrkt_c_api_compile_check(void);

void torchrkt_c_api_compile_check(void) {
  const char* (*version)(void) = tr_version;
  const char* (*last_error)(void) = tr_last_error;
  int (*last_error_kind)(void) = tr_last_error_kind;
  int (*manual_seed)(uint64_t) = tr_manual_seed;
  tr_tensor* (*randn)(const int64_t*, int64_t) = tr_randn;
  void (*tensor_free)(tr_tensor*) = tr_tensor_free;
  int (*numel)(const tr_tensor*, int64_t*) = tr_tensor_numel;
  int (*nbytes)(const tr_tensor*, int64_t*) = tr_tensor_nbytes;
  int (*copy_data)(const tr_tensor*, uint64_t, float*, uint64_t*) =
      tr_tensor_copy_data;
  tr_tensor* (*zeros)(const int64_t*, int64_t) = tr_zeros;
  tr_tensor* (*from_data)(const float*, uint64_t, const int64_t*, int64_t) =
      tr_from_data;
  tr_tensor* (*from_data_on)(const float*, uint64_t, const int64_t*, int64_t,
                             tr_device_type, int64_t) = tr_from_data_on_device;
  tr_tensor* (*from_data_i64)(const int64_t*, uint64_t, const int64_t*,
                              int64_t) = tr_from_data_i64;
  tr_tensor* (*from_data_i64_on)(const int64_t*, uint64_t, const int64_t*,
                                 int64_t, tr_device_type, int64_t) =
      tr_from_data_i64_on_device;
  int (*tensor_dtype)(const tr_tensor*, tr_dtype*) = tr_tensor_dtype;
  int (*copy_data_i64)(const tr_tensor*, uint64_t, int64_t*, uint64_t*) =
      tr_tensor_copy_data_i64;
  int (*copy_data_f64)(const tr_tensor*, uint64_t, double*, uint64_t*) =
      tr_tensor_copy_data_f64;
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
  /* Generated surface: one pin per generated signature shape. */
  tr_tensor* (*gen_matmul)(const tr_tensor*, const tr_tensor*) = tr_gen_matmul;
  tr_tensor* (*gen_reshape)(const tr_tensor*, const int64_t*, int64_t) =
      tr_gen_reshape;
  tr_tensor* (*gen_cat)(const tr_tensor* const*, int64_t, int64_t) = tr_gen_cat;
  int (*gen_mul_)(tr_tensor*, const tr_tensor*) = tr_gen_mul__tensor;
  int (*gen_add_)(tr_tensor*, const tr_tensor*, double) = tr_gen_add__tensor;
  int (*gen_lerp_)(tr_tensor*, const tr_tensor*, const tr_tensor*) =
      tr_gen_lerp__tensor;
  int (*gen_addcmul_)(tr_tensor*, const tr_tensor*, const tr_tensor*, double) =
      tr_gen_addcmul_;
  int (*gen_copy_)(tr_tensor*, const tr_tensor*, bool) = tr_gen_copy_;
  int (*gen_fill_)(tr_tensor*, double) = tr_gen_fill__scalar;
  int (*gen_index_copy_)(tr_tensor*, int64_t, const tr_tensor*,
                         const tr_tensor*) = tr_gen_index_copy_;
  int (*gen_index_add_)(tr_tensor*, int64_t, const tr_tensor*, const tr_tensor*,
                        double) = tr_gen_index_add_;
  int (*gen_scatter_value_)(tr_tensor*, int64_t, const tr_tensor*, double) =
      tr_gen_scatter__value;
  int (*gen_masked_fill_)(tr_tensor*, const tr_tensor*, double) =
      tr_gen_masked_fill__scalar;
  int (*gen_index_fill_scalar_)(tr_tensor*, int64_t, const tr_tensor*, double) =
      tr_gen_index_fill__int_scalar;
  int (*gen_index_fill_tensor_)(tr_tensor*, int64_t, const tr_tensor*,
                                const tr_tensor*) =
      tr_gen_index_fill__int_tensor;
  int (*gen_masked_fill_tensor_)(tr_tensor*, const tr_tensor*,
                                 const tr_tensor*) = tr_gen_masked_fill__tensor;
  int (*gen_masked_scatter_)(tr_tensor*, const tr_tensor*, const tr_tensor*) =
      tr_gen_masked_scatter_;
  int (*gen_scatter_src_)(tr_tensor*, int64_t, const tr_tensor*,
                          const tr_tensor*) = tr_gen_scatter__src;
  int (*gen_scatter_add_)(tr_tensor*, int64_t, const tr_tensor*,
                          const tr_tensor*) = tr_gen_scatter_add_;
  tr_tensor* (*gen_sum_dim)(const tr_tensor*, const int64_t*, int64_t, bool,
                            bool, int32_t) = tr_gen_sum_dim_intlist;
  tr_tensor* (*gen_conv2d)(const tr_tensor*, const tr_tensor*, const tr_tensor*,
                           const int64_t*, int64_t, const int64_t*, int64_t,
                           const int64_t*, int64_t, int64_t) = tr_gen_conv2d;
  tr_tensor* (*gen_avg_pool2d)(const tr_tensor*, const int64_t*, int64_t,
                               const int64_t*, int64_t, const int64_t*, int64_t,
                               bool, bool, int64_t, bool) = tr_gen_avg_pool2d;
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
  tr_tensor* (*gen_narrow)(const tr_tensor*, int64_t, int64_t, int64_t) =
      tr_gen_narrow;
  tr_tensor* (*gen_select)(const tr_tensor*, int64_t, int64_t) =
      tr_gen_select_int;
  tr_tensor* (*gen_slice)(const tr_tensor*, int64_t, int64_t, bool, int64_t,
                          bool, int64_t) = tr_gen_slice_tensor;
  tr_tensor* (*gen_index_select)(const tr_tensor*, int64_t, const tr_tensor*) =
      tr_gen_index_select;
  tr_tensor* (*gen_masked_select)(const tr_tensor*, const tr_tensor*) =
      tr_gen_masked_select;
  tr_tensor* (*gen_nonzero)(const tr_tensor*) = tr_gen_nonzero;
  tr_tensor* (*gen_take)(const tr_tensor*, const tr_tensor*) = tr_gen_take;
  tr_tensor* (*gen_gather)(const tr_tensor*, int64_t, const tr_tensor*, bool) =
      tr_gen_gather;
  tr_tensor* (*gen_take_along_dim)(const tr_tensor*, const tr_tensor*, int64_t,
                                   bool) = tr_gen_take_along_dim;
  tr_tensor* (*gen_where_self)(const tr_tensor*, const tr_tensor*,
                               const tr_tensor*) = tr_gen_where_self;
  tr_tensor* (*gen_where_scalarother)(const tr_tensor*, const tr_tensor*,
                                      double) = tr_gen_where_scalarother;
  tr_tensor* (*gen_where_scalarself)(
      const tr_tensor*, double, const tr_tensor*) = tr_gen_where_scalarself;
  tr_tensor* (*gen_where_scalar)(const tr_tensor*, double, double) =
      tr_gen_where_scalar;
  tr_tensor* (*gen_dropout)(const tr_tensor*, double, bool) = tr_gen_dropout;
  int (*cuda_available)(void) = tr_cuda_is_available;
  int (*cuda_count)(void) = tr_cuda_device_count;
  int (*cuda_stats)(int64_t, int64_t*, int64_t*, int64_t*) =
      tr_cuda_memory_stats;
  int (*cuda_empty)(void) = tr_cuda_empty_cache;
  int (*mps_available)(void) = tr_mps_is_available;
  int (*mps_empty)(void) = tr_mps_empty_cache;
  int (*set_default_device)(tr_device_type, int64_t) = tr_set_default_device;
  int (*get_default_device)(tr_device_type*, int64_t*) = tr_get_default_device;
  tr_tensor* (*to_device)(const tr_tensor*, tr_device_type, int64_t) =
      tr_tensor_to_device;
  int (*tensor_device)(const tr_tensor*, tr_device_type*, int64_t*) =
      tr_tensor_device;

  (void)version;
  (void)last_error;
  (void)last_error_kind;
  (void)manual_seed;
  (void)randn;
  (void)tensor_free;
  (void)numel;
  (void)nbytes;
  (void)copy_data;
  (void)zeros;
  (void)from_data;
  (void)from_data_i64;
  (void)from_data_i64_on;
  (void)tensor_dtype;
  (void)copy_data_i64;
  (void)copy_data_f64;
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
  (void)gen_fill_;
  (void)gen_index_copy_;
  (void)gen_index_add_;
  (void)gen_scatter_value_;
  (void)gen_masked_fill_;
  (void)gen_index_fill_scalar_;
  (void)gen_index_fill_tensor_;
  (void)gen_masked_fill_tensor_;
  (void)gen_masked_scatter_;
  (void)gen_scatter_src_;
  (void)gen_scatter_add_;
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
  (void)gen_select;
  (void)gen_slice;
  (void)gen_index_select;
  (void)gen_masked_select;
  (void)gen_nonzero;
  (void)gen_take;
  (void)gen_gather;
  (void)gen_take_along_dim;
  (void)gen_where_self;
  (void)gen_where_scalarother;
  (void)gen_where_scalarself;
  (void)gen_where_scalar;
  (void)gen_dropout;
  (void)cuda_available;
  (void)cuda_count;
  (void)cuda_stats;
  (void)cuda_empty;
  (void)mps_available;
  (void)mps_empty;
  (void)set_default_device;
  (void)get_default_device;
  (void)to_device;
  (void)tensor_device;
}

#pragma once

#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_add(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_sub(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_mul(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_div(const tr_tensor* a, const tr_tensor* b);
tr_tensor* tr_pow(const tr_tensor* a, const tr_tensor* b);

tr_tensor* tr_add_scalar(const tr_tensor* a, double b);
tr_tensor* tr_sub_scalar(const tr_tensor* a, double b);
tr_tensor* tr_mul_scalar(const tr_tensor* a, double b);
tr_tensor* tr_div_scalar(const tr_tensor* a, double b);
tr_tensor* tr_pow_scalar(const tr_tensor* a, double b);

tr_tensor* tr_abs(const tr_tensor* t);
tr_tensor* tr_neg(const tr_tensor* t);
tr_tensor* tr_exp(const tr_tensor* t);
tr_tensor* tr_log(const tr_tensor* t);
tr_tensor* tr_sqrt(const tr_tensor* t);
tr_tensor* tr_relu(const tr_tensor* t);
tr_tensor* tr_sigmoid(const tr_tensor* t);
tr_tensor* tr_tanh(const tr_tensor* t);
tr_tensor* tr_gelu(const tr_tensor* t);

#ifdef __cplusplus
}
#endif

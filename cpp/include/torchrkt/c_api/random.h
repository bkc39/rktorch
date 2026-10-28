#pragma once

#include <stdint.h>

#include "torchrkt/c_api/device.h"
#include "torchrkt/c_api/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

tr_tensor* tr_randn(const int64_t* dims, int64_t ndim);

tr_tensor* tr_rand(const int64_t* dims, int64_t ndim);

/* torch.randn / torch.rand with the device and dtype chosen at construction
 * (TR_DEVICE_KEEP / TR_DTYPE_KEEP: the process default device / float32);
 * both draw from the chosen device's generator. */
tr_tensor* tr_randn_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                       int64_t index, tr_dtype dtype);

tr_tensor* tr_rand_on(const int64_t* dims, int64_t ndim, tr_device_type type,
                      int64_t index, tr_dtype dtype);

/* A CPU random generator with its own stream (torch.Generator). A data
 * loader's shuffle draws from one so a seeded permutation replays
 * independently of the global stream. Release with tr_generator_free (safe
 * on NULL); NULL on error. */
typedef struct tr_generator tr_generator;

/* tr_randn_on / tr_rand_on with torch.randn / torch.rand(generator=g): they
 * draw from g, or from the chosen device's default generator when g is NULL.
 * A CPU g with a non-CPU device fails cleanly. */
tr_tensor* tr_randn_gen_on(const int64_t* dims, int64_t ndim,
                           tr_device_type type, int64_t index, tr_dtype dtype,
                           tr_generator* g);

tr_tensor* tr_rand_gen_on(const int64_t* dims, int64_t ndim,
                          tr_device_type type, int64_t index, tr_dtype dtype,
                          tr_generator* g);

tr_generator* tr_generator_new(uint64_t seed);

/* t.uniform_(low, high, generator=g) / t.normal_(mean, std, generator=g): in
 * place, from g, or from t's device's default generator when g is NULL. A g
 * on another device than t fails cleanly. 0 success, 1 error. */
int tr_tensor_uniform_(tr_tensor* t, double low, double high, tr_generator* g);

int tr_tensor_normal_(tr_tensor* t, double mean, double std, tr_generator* g);

/* torch.Generator(device).manual_seed(seed): a generator with its own stream
 * on the given device, drawing as that device's default generator does.
 * NULL on error (unknown device, an index out of range, or a backend that is
 * not built in). */
tr_generator* tr_generator_new_on(uint64_t seed, tr_device_type type,
                                  int64_t index);

/* The device a generator draws on. 0 success, 1 error. */
int tr_generator_device(const tr_generator* g, tr_device_type* out_type,
                        int64_t* out_index);

void tr_generator_free(tr_generator* g);

/* torch.randperm(n, generator=g): an int64 permutation of 0..n-1 on the CPU
 * drawn from g, or from the global CPU generator when g is NULL. */
tr_tensor* tr_randperm(int64_t n, tr_generator* g);

/* torch.empty((), dtype=torch.int64).random_(generator=g).item(): one int64
 * drawn from g on g's device, or from the global CPU generator when g is
 * NULL. A DataLoader draws one per epoch before its permutation, so a loader
 * that replays its batch order needs the same draw. 0 success, 1 error. */
int tr_generator_draw_seed(tr_generator* g, int64_t* out);

#ifdef __cplusplus
}
#endif

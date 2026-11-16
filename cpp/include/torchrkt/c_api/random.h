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

int tr_tensor_uniform_(tr_tensor* t, double low, double high);

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
 * place, from g, or from t's device's default generator when g is NULL
 * (tr_tensor_uniform_ is tr_tensor_uniform_gen_ with g NULL). A g on another
 * device than t fails cleanly. 0 success, 1 error. */
int tr_tensor_uniform_gen_(tr_tensor* t, double low, double high,
                           tr_generator* g);

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

/* g.get_state(): a copy of g's state, a CPU uint8 tensor (5056 bytes for a
 * CPU generator, 16 for CUDA: seed then Philox offset). NULL on error. */
tr_tensor* tr_generator_state(const tr_generator* g);

/* g.set_state(state): g continues from a copy of state. 0 success, 1 error
 * (libtorch's own size/type/mt19937 checks; the shim adds none). */
int tr_generator_set_state(tr_generator* g, const tr_tensor* state);

/* torch.get_rng_state() / torch.cuda.get_rng_state(index): the device's
 * default generator, the one tr_manual_seed seeds. NULL on error (unknown
 * device, an index out of range, a backend that is not built in). */
tr_tensor* tr_default_generator_state(tr_device_type type, int64_t index);

/* torch.set_rng_state(state) / torch.cuda.set_rng_state(state, index), as
 * tr_generator_set_state on the device's default generator. */
int tr_default_generator_set_state(tr_device_type type, int64_t index,
                                   const tr_tensor* state);

#ifdef __cplusplus
}
#endif

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* libtorch version as a "MAJOR.MINOR.PATCH" UTF-8 string. */
const char* tr_version(void);

/* Last error message set by a failing tr_* call (thread-local). */
const char* tr_last_error(void);

/* Machine-readable kind of the last error on this thread, paired with
 * tr_last_error (every error record updates both): 0 generic, 1
 * out-of-memory (CUDA's c10::OutOfMemoryError, or the CPU allocator's
 * enforce-path failure). */
int tr_last_error_kind(void);

/* Seed the global ATen CPU RNG. 0: success, 1: error (see tr_last_error). */
int tr_manual_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

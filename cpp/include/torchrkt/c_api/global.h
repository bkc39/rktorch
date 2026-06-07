#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* libtorch version as a "MAJOR.MINOR.PATCH" UTF-8 string. */
const char* tr_version(void);

/* Last error message set by a failing tr_* call (thread-local). */
const char* tr_last_error(void);

/* Seed the global ATen CPU RNG. 0: success, 1: error (see tr_last_error). */
int tr_manual_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

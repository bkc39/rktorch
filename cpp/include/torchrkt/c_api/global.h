#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* tr_version(void);

/* SHA-256 of the c_api headers this shim was built from. */
const char* tr_abi_stamp(void);

/* Thread-local, as is tr_last_error_kind. */
const char* tr_last_error(void);

/* 0 generic, 1 out-of-memory. */
int tr_last_error_kind(void);

int tr_manual_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

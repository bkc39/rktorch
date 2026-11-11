/* Compile-only: pins C linkage for the global.h entry points. */
#include <stdint.h>

#include "torchrkt/c_api.h"

void torchrkt_c_api_compile_global_check(void);

void torchrkt_c_api_compile_global_check(void) {
  const char* (*version)(void) = tr_version;
  const char* (*abi_stamp)(void) = tr_abi_stamp;
  const char* (*last_error)(void) = tr_last_error;
  int (*last_error_kind)(void) = tr_last_error_kind;
  int (*manual_seed)(uint64_t) = tr_manual_seed;
  int (*set_deterministic_algorithms)(int, int) =
      tr_set_deterministic_algorithms;
  int (*get_deterministic_algorithms)(int*) = tr_get_deterministic_algorithms;
  int (*get_deterministic_algorithms_warn_only)(int*) =
      tr_get_deterministic_algorithms_warn_only;
  int (*set_cudnn_deterministic)(int) = tr_set_cudnn_deterministic;
  int (*get_cudnn_deterministic)(int*) = tr_get_cudnn_deterministic;
  int (*set_cudnn_benchmark)(int) = tr_set_cudnn_benchmark;
  int (*get_cudnn_benchmark)(int*) = tr_get_cudnn_benchmark;

  (void)version;
  (void)abi_stamp;
  (void)last_error;
  (void)last_error_kind;
  (void)manual_seed;
  (void)set_deterministic_algorithms;
  (void)get_deterministic_algorithms;
  (void)get_deterministic_algorithms_warn_only;
  (void)set_cudnn_deterministic;
  (void)get_cudnn_deterministic;
  (void)set_cudnn_benchmark;
  (void)get_cudnn_benchmark;
}

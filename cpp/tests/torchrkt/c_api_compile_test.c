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

  (void)version;
  (void)last_error;
  (void)manual_seed;
  (void)randn;
  (void)tensor_free;
  (void)numel;
  (void)copy_data;
}

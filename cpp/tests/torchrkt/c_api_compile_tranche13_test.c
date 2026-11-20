/* Compile-only: pins C linkage for generated tranche 13. */
#include <stdbool.h>
#include <stdint.h>

#include "torchrkt/c_api.h"

void torchrkt_c_api_compile_tranche13_check(void);

void torchrkt_c_api_compile_tranche13_check(void) {
  tr_tensor* (*gen_bilinear)(const tr_tensor*, const int64_t*, int64_t, bool,
                             double, bool, double, bool) =
      tr_gen_upsample_bilinear2d;
  tr_tensor* (*gen_bilinear_aa)(const tr_tensor*, const int64_t*, int64_t, bool,
                                double, bool, double, bool) =
      tr_gen__upsample_bilinear2d_aa;
  tr_tensor* (*gen_round)(const tr_tensor*) = tr_gen_round;
  (void)gen_bilinear;
  (void)gen_bilinear_aa;
  (void)gen_round;
}

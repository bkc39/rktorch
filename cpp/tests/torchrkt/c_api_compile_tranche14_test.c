/* Compile-only: pins C linkage for generated tranche 14. */
#include "torchrkt/c_api.h"

void torchrkt_c_api_compile_tranche14_check(void);

void torchrkt_c_api_compile_tranche14_check(void) {
  int (*gen_div_tensor_)(tr_tensor*, const tr_tensor*) = tr_gen_div__tensor;
  (void)gen_div_tensor_;
}

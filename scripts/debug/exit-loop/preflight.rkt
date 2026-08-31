#lang racket/base
(require (only-in racket/system system)
         (only-in torch cuda-available? cuda-device-count cuda-memory-stats
                  cuda-device finalizer-failures mps-available? native-memory-use
                  ones tensor-shape))

(printf "racket        : ~a ~a\n" (version) (system-type 'machine))
(printf "os            : ~a / ~a\n" (system-type 'os*) (system-type 'arch))
(printf "cuda-available: ~a\n" (cuda-available?))
(when (cuda-available?)
  (printf "cuda devices  : ~a\n" (cuda-device-count))
  (printf "cuda stats    : ~a\n" (cuda-memory-stats (cuda-device 0))))
(printf "mps-available : ~a\n" (mps-available?))
(printf "smoke (ones)  : ~a\n" (tensor-shape (ones 2 2)))
(printf "ledger        : ~a\n" (native-memory-use))
(printf "fin-failures  : ~a\n" (finalizer-failures))
(newline)
(display "staged native lib:\n")
(void (system "ls -l torch/native-libs/ 2>/dev/null || true"))
(display "\nlinkage:\n")
(void (system (if (eq? (system-type 'os) 'macosx)
                  "otool -L torch/native-libs/libtorchrkt.dylib 2>/dev/null | head -8"
                  "ldd torch/native-libs/libtorchrkt.so 2>/dev/null | head -12")))

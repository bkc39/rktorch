#lang info

;; raco review lints this as a normal module and flags every `info` definition
;; as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

(define collection "torchrkt")
(define version "0.1")
(define deps '("base"))
(define build-deps '("rackunit-lib" "racket-doc" "scribble-lib"))
(define pkg-desc "Racket bindings for libtorch (PyTorch)")
(define pkg-authors '("bkschemer@gmail.com"))
(define license 'Apache-2.0)
(define pkg-tags
  '("machine-learning" "deep-learning" "tensor" "pytorch" "libtorch"))
(define pre-install-collection "private/install-torchrkt-native.rkt")

;; `raco test --drdr` defaults to a 90s per-test timeout; the native-backed
;; tests get generous headroom so a loaded builder cannot transiently time out.
(define test-timeouts
  '(("tests/foreign-test.rkt" 300)))

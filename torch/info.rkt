#lang info

;; raco review lints this as a normal module and flags every `info` definition
;; as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

(define collection "torch")
(define version "0.1")
(define deps
  '(("base" #:version "9.1")
    "threading-lib"
    ("torch-x86_64-linux" #:platform "x86_64-linux" #:version "0.1")))
(define build-deps
  '("at-exp-lib"
    "draw-lib"
    "rackunit-lib"
    "racket-doc"
    "sandbox-lib"
    "scribble-lib"
    "threading-doc"))
(define scribblings (quote (("scribblings/torch.scrbl" (multi-page)))))
(define pkg-desc "Racket bindings for libtorch (PyTorch)")
(define pkg-authors (list (string-append "bkschemer" "@" "gmail" "." "com")))
(define license '(Apache-2.0 AND CC-BY-4.0 AND CC-BY-2.0-FR))
(define pkg-tags
  '("machine-learning" "deep-learning" "tensor" "pytorch" "libtorch"))
(define pre-install-collection "private/install-torchrkt-native.rkt")

;; Enforced under `raco test --timeout` (the flake's check passes 300s as the
;; default). The parity suites self-skip without python3 `torch` and take
;; minutes with it.
(define test-timeouts
  '(("tests/finalizer-thread-test.rkt" 120)
    ("tests/foreign-test.rkt" 300)
    ("tests/generated-parity-test.rkt" 900)
    ("tests/python-cross-test.rkt" 1200)))

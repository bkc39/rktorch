#lang info

#|review: ignore|#

(define collection "rktorch-lint")
(define version "0.1")
(define deps '("base" "bkc-style" "resyntax" "torch"))
(define pkg-desc "Resyntax suites for the torch bindings: the gating suite and the advisory candidates")
(define license 'Apache-2.0)
(define test-include-paths '("tests/torch-forms-test.resyntax"))

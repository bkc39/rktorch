#lang racket/base

;; raco review does surface-level linting without macro expansion, so it cannot
;; see that every `contract-out` entry below re-exports an imported identifier —
;; it would report each as "provided but not defined".  This is a pure
;; re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the safe, contracted FFI layer.  `(require torchrkt/foreign)`
;; exposes the low-level surface; all implementation lives in foreign/*.  A
;; `tensor` is a wrapper struct over a cpointer whose underlying handle is
;; GC-reclaimed.
;;
;; `(require (submod torchrkt/foreign unsafe))` additionally exposes
;; `tensor-free!` for callers that need deterministic release.  It is
;; idempotent: a second call hits the cpointer tag guard and raises
;; `exn:fail:contract` instead of double-freeing.
;;
;; Contracts are applied here, at the facade boundary, so this file is the
;; single authoritative description of the public surface.

(require ffi/vector
         racket/contract
         "foreign/structs.rkt"
         "foreign/ops.rkt")

(provide
 (contract-out
  [torch-version (-> string?)]
  [manual-seed! (-> exact-nonnegative-integer? void?)]
  [randn (->* () #:rest (listof exact-nonnegative-integer?) tensor?)]
  [tensor? (-> any/c boolean?)]
  [tensor-shape (-> tensor? (listof exact-nonnegative-integer?))]
  [tensor-numel (-> tensor? exact-nonnegative-integer?)]
  [tensor->vector (-> tensor? f32vector?)]
  [tensor->list (-> tensor? (listof real?))]
  ;; tensor->repr: the PyTorch `repr` text (what the REPL prints);
  ;; tensor->string: ATen's C++ `operator<<` text.
  [tensor->repr (-> tensor? string?)]
  [tensor->string (-> tensor? string?)]))

;; Explicit-free helper for deterministic release.  See the file comment.
(module+ unsafe
  (provide
   (contract-out
    [tensor-free! (-> tensor? void?)])))

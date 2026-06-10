#lang racket/base

;; Safe autograd surface: requires-grad / backward / grad / detach, the
;; with-no-grad dynamic extent, and the in-place ops the optimizer uses.
;; Contracts live in ../foreign.rkt (with-no-grad is a macro, exported as-is).

(require "error.rkt"
         "raw/autograd.rkt"
         "structs.rkt")

(provide requires-grad!
         requires-grad?
         backward!
         grad
         has-grad?
         maybe-grad
         detach
         grad-enabled?
         call-with-no-grad
         with-no-grad
         sub!
         zero!
         mul!
         zero-grad!)

;; Flip requires_grad in place; returns the tensor so it chains in
;; constructors: (requires-grad! (randn 2 2) #t).
(define (requires-grad! t [on? #t])
  (check-ok (tr-tensor-requires-grad!/raw t on?) 'requires-grad!)
  t)

(define (requires-grad? t)
  (define-values (rc on?) (tr-tensor-requires-grad/raw t))
  (check-ok rc 'requires-grad?)
  on?)

(define (backward! t)
  (check-ok (tr-tensor-backward/raw t) 'backward!)
  (void))

;; The accumulated gradient. Shares storage with the live .grad, so (zero!
;; (grad t)) really zeroes the gradient backward will next accumulate into.
(define (grad t)
  (wrap-tensor (check-handle 'grad (tr-tensor-grad/raw t))))

;; #t once backward has accumulated a gradient (PyTorch: grad is not None).
;; Uses the dedicated C predicate: no handle allocation and no stale
;; tr_last_error on the "no gradient" path.
(define (has-grad? t)
  (define-values (rc on?) (tr-tensor-has-grad/raw t))
  (check-ok rc 'has-grad?)
  on?)

;; The gradient, or #f if none has been accumulated yet — the optimizer's
;; single-allocation path (has-grad? + grad would allocate the handle and
;; walk the FFI twice per parameter per step).
(define (maybe-grad t)
  (and (has-grad? t) (grad t)))

(define (detach t)
  (wrap-tensor (check-handle 'detach (tr-tensor-detach/raw t))))

(define (grad-enabled?)
  (define-values (rc on?) (tr-is-grad-enabled/raw))
  (check-ok rc 'grad-enabled?)
  on?)

(define (set-grad-enabled! on?)
  (check-ok (tr-set-grad-enabled/raw on?) 'set-grad-enabled!))

;; Grad mode is thread-local on the C++ side and every FFI call happens on
;; Racket's OS thread, so dynamic-wind gives torch.no_grad() semantics
;; (restored on escape, continuation re-entry re-disables).
(define (call-with-no-grad thunk)
  (define was? (grad-enabled?))
  (dynamic-wind (lambda () (set-grad-enabled! #f))
                thunk
                (lambda () (set-grad-enabled! was?))))

(define-syntax-rule (with-no-grad body ...)
  (call-with-no-grad (lambda () body ...)))

;; In-place update primitives (use under with-no-grad on parameters, exactly
;; like torch.optim does).
(define (sub! t other [alpha 1.0])
  (check-ok (tr-tensor-sub!/raw t other (exact->inexact alpha)) 'sub!)
  (void))

(define (zero! t)
  (check-ok (tr-tensor-zero!/raw t) 'zero!)
  (void))

(define (mul! t value)
  (check-ok (tr-tensor-mul!/raw t (exact->inexact value)) 'mul!)
  (void))

;; Zero a parameter's accumulated gradient (optimizer.zero_grad for one
;; tensor). A tensor that never ran backward has no grad; treat as a no-op so
;; (zero-grad!) is safe before the first step.
(define (zero-grad! t)
  (define h (tr-tensor-grad/raw t))
  (when h
    (zero! (wrap-tensor h))))

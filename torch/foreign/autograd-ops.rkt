#lang racket/base

(require (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/autograd.rkt"
                  tr-is-grad-enabled/raw
                  tr-set-grad-enabled/raw
                  tr-tensor-backward/raw
                  tr-tensor-detach/raw
                  tr-tensor-grad/raw
                  tr-tensor-has-grad/raw
                  tr-tensor-mul!/raw
                  tr-tensor-requires-grad!/raw
                  tr-tensor-requires-grad/raw
                  tr-tensor-sub!/raw
                  tr-tensor-zero!/raw)
         (only-in "structs.rkt" wrap-tensor))

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

(define (grad t)
  (wrap-tensor (check-handle 'grad (tr-tensor-grad/raw t))))

(define (has-grad? t)
  (define-values (rc on?) (tr-tensor-has-grad/raw t))
  (check-ok rc 'has-grad?)
  on?)

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

(define (call-with-no-grad thunk)
  (define was? (grad-enabled?))
  (dynamic-wind (lambda () (set-grad-enabled! #f))
                thunk
                (lambda () (set-grad-enabled! was?))))

(define-syntax-rule (with-no-grad body ...)
  (call-with-no-grad (lambda () body ...)))

(define (sub! t other [alpha 1.0])
  (check-ok (tr-tensor-sub!/raw t other (exact->inexact alpha)) 'sub!)
  (void))

(define (zero! t)
  (check-ok (tr-tensor-zero!/raw t) 'zero!)
  (void))

(define (mul! t value)
  (check-ok (tr-tensor-mul!/raw t (exact->inexact value)) 'mul!)
  (void))

(define (zero-grad! t)
  (when (has-grad? t)
    (zero! (grad t))))

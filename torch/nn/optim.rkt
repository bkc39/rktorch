#lang racket/base

;; Plain SGD over a parameter list (what (parameters model) returns). The
;; update is pure Racket over the in-place primitives, run under
;; with-no-grad exactly like torch.optim: p -= lr * p.grad.

(require (only-in "../foreign.rkt" maybe-grad sub! with-no-grad zero-grad!))

(provide sgd
         sgd?
         sgd-lr
         step!
         zero-grads!)

;; sgd-optimizer takes the static-info binding so the `sgd` name is free for
;; the keyword smart constructor below.
(struct sgd (params lr)
  #:constructor-name make-sgd
  #:name sgd-optimizer) ;; noqa

(define (sgd params #:lr lr)
  (make-sgd params lr))

;; One update step. Parameters that never received a gradient are skipped
;; (PyTorch skips grad-is-None parameters the same way). maybe-grad keeps it
;; to one native grad-handle allocation per parameter per step.
(define (step! opt)
  (with-no-grad
    (for ([p (in-list (sgd-params opt))])
      (define g (maybe-grad p))
      (when g
        (sub! p g (sgd-lr opt))))))

;; optimizer.zero_grad(): reset every accumulated gradient to zero.
(define (zero-grads! opt)
  (for-each zero-grad! (sgd-params opt)))

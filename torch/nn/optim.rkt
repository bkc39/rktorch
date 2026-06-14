#lang racket/base

;; Optimizers over a parameter list (what (parameters model) returns). The
;; updates run under with-no-grad exactly like torch.optim and mutate each
;; parameter in place; the per-step math is pure Racket over the shadow
;; arithmetic operators (so number*tensor is a scalar scale, tensor*tensor is
;; elementwise, etc.).

(require (only-in racket/generic define-generics)
         (only-in "../foreign.rkt"
                  + - * / sqrt
                  maybe-grad sub! tensor-shape with-no-grad zero-grad! zeros))

(provide gen:optimizer
         optimizer?
         sgd
         sgd?
         sgd-lr
         adam
         adam?
         step!
         zero-grads!)

;; The optimizer interface: one update step + the parameter list (so
;; zero-grads! is shared across optimizers).
(define-generics optimizer
  (optimizer-step! optimizer)
  (optimizer-parameters optimizer))

;; --- SGD ----------------------------------------------------------------
;; p -= lr * p.grad. Parameters that never received a gradient are skipped
;; (PyTorch skips grad-is-None params the same way).
(struct sgd (params lr)
  #:constructor-name make-sgd
  #:name sgd-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (sgd-params opt))
   (define (optimizer-step! opt)
     (with-no-grad
       (for ([p (in-list (sgd-params opt))])
         (define g (maybe-grad p))
         (when g
           (sub! p g (sgd-lr opt))))))])

(define (sgd params #:lr lr)
  (make-sgd params lr))

;; --- Adam ---------------------------------------------------------------
;; torch.optim.Adam: per-parameter first/second moment EMAs with bias
;; correction. Moments are lazily allocated (zeros) the first time a parameter
;; is seen and kept in eq?-hashes keyed by the parameter tensor.
(struct adam (params lr beta1 beta2 eps step-box m v)
  #:constructor-name make-adam
  #:name adam-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (adam-params opt))
   (define (optimizer-step! opt) (adam-do-step! opt))])

(define (adam params
              #:lr [lr 1e-3]
              #:beta1 [beta1 0.9]
              #:beta2 [beta2 0.999]
              #:eps [eps 1e-8])
  (make-adam params lr beta1 beta2 eps (box 0) (make-hasheq) (make-hasheq)))

(define (zeros-like t)
  (apply zeros (tensor-shape t)))

(define (adam-do-step! opt)
  (with-no-grad
    (set-box! (adam-step-box opt) (add1 (unbox (adam-step-box opt))))
    (define t (unbox (adam-step-box opt)))
    (define b1 (adam-beta1 opt))
    (define b2 (adam-beta2 opt))
    (define lr (adam-lr opt))
    (define eps (adam-eps opt))
    ;; bias corrections (plain numbers)
    (define bc1 (- 1.0 (expt b1 t)))
    (define bc2 (- 1.0 (expt b2 t)))
    (for ([p (in-list (adam-params opt))])
      (define g (maybe-grad p))
      (when g
        (define m (hash-ref! (adam-m opt) p (lambda () (zeros-like p))))
        (define v (hash-ref! (adam-v opt) p (lambda () (zeros-like p))))
        ;; m <- b1*m + (1-b1)*g ;  v <- b2*v + (1-b2)*g^2
        (define m* (+ (* b1 m) (* (- 1.0 b1) g)))
        (define v* (+ (* b2 v) (* (- 1.0 b2) (* g g))))
        (hash-set! (adam-m opt) p m*)
        (hash-set! (adam-v opt) p v*)
        ;; p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)  (torch.optim.Adam)
        (define denom (+ (sqrt (/ v* bc2)) eps))
        (sub! p (/ (/ m* bc1) denom) lr)))))

;; --- shared surface -----------------------------------------------------
(define (step! opt)
  (optimizer-step! opt))

(define (zero-grads! opt)
  (for-each zero-grad! (optimizer-parameters opt)))

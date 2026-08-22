#lang racket/base

;; Optimizers over a parameter list. Updates run under with-no-grad and
;; mutate in place, like torch.optim; the step math uses foreign.rkt's shadow
;; arithmetic operators (number*tensor scales, tensor*tensor is elementwise).

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

;; parameters are part of the interface so zero-grads! is shared.
(define-generics optimizer
  (optimizer-step! optimizer)
  (optimizer-parameters optimizer))

;; --- SGD ----------------------------------------------------------------
;; Parameters that never received a gradient are skipped, mirroring how
;; PyTorch skips grad-is-None params.
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
;; torch.optim.Adam (moment EMAs + bias correction). Moments are lazily
;; allocated per parameter, in eq?-hashes keyed by the parameter tensor.
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
    (define bc1 (- 1.0 (expt b1 t)))
    (define bc2 (- 1.0 (expt b2 t)))
    (for ([p (in-list (adam-params opt))])
      (define g (maybe-grad p))
      (when g
        (define m (hash-ref! (adam-m opt) p (lambda () (zeros-like p))))
        (define v (hash-ref! (adam-v opt) p (lambda () (zeros-like p))))
        (define m* (+ (* b1 m) (* (- 1.0 b1) g)))
        (define v* (+ (* b2 v) (* (- 1.0 b2) (* g g))))
        (hash-set! (adam-m opt) p m*)
        (hash-set! (adam-v opt) p v*)
        (define denom (+ (sqrt (/ v* bc2)) eps))
        (sub! p (/ (/ m* bc1) denom) lr)))))

;; --- shared surface -----------------------------------------------------
(define (step! opt)
  (optimizer-step! opt))

(define (zero-grads! opt)
  (for-each zero-grad! (optimizer-parameters opt)))

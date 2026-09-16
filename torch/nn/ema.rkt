#lang racket/base

(require (only-in racket/contract/base -> ->i any/c contract-out real-in)
         (only-in "../foreign.rkt"
                  copy! dtype full lerp! shape tensor-device with-no-grad)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" layer? parameters))

(struct ema (model average decay count weights) ;; noqa
  #:constructor-name make-ema
  #:omit-define-syntaxes)

(provide (contract-out
          [ema? (-> any/c boolean?)]
          [ema-average (-> ema? layer?)]
          [ema-decay (-> ema? (real-in 0 1))]))

(define (separate-copy? model average)
  (define ps (parameters model))
  (define qs (parameters average))
  (and (not (eq? model average))
       (= (length ps) (length qs))
       (for/and ([p (in-list ps)] [q (in-list qs)])
         (and (not (memq q ps))
              (equal? (shape p) (shape q))
              (equal? (tensor-device p) (tensor-device q))
              (eq? (dtype p) (dtype q))))))

(define (copy-parameters! e)
  (with-no-grad
    (for ([p (in-list (parameters (ema-model e)))]
          [q (in-list (parameters (ema-average e)))])
      (copy! q p))))

(define/contract-out (ema model average #:decay [decay 0.9999]) ;; noqa
  (->i ([model layer?] [average layer?])
       (#:decay [decay (real-in 0 1)])
       #:pre/name (model average)
       "the average must be a separate layer with the model's parameters, shape for shape, on its device and dtype"
       (separate-copy? model average)
       [result ema?])
  (define e (make-ema model average decay (box 0) (make-hasheq)))
  (copy-parameters! e)
  e)

(define/contract-out (ema-update! e) ;; noqa
  (-> ema? void?)
  (define n (unbox (ema-count e)))
  (set-box! (ema-count e) (add1 n))
  (cond
    [(zero? n) (copy-parameters! e)]
    [else
     (with-no-grad
       (for ([p (in-list (parameters (ema-model e)))]
             [q (in-list (parameters (ema-average e)))])
         (lerp! q p (weight-on e q))))]))

(define (weight-on e q)
  (hash-ref! (ema-weights e) q
             (lambda ()
               (full (- 1.0 (ema-decay e)) #:device (tensor-device q) #:dtype (dtype q)))))

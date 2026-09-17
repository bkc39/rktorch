#lang racket/base

(require (only-in racket/contract/base -> ->* any or/c)
         syntax/parse/define
         (prefix-in g: (only-in "../generated.rkt"
                                addcdiv! addcmul! copy! lerp-tensor!))
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "error.rkt" check-handle check-ok)
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
         (only-in "raw/memory.rkt" collect-at-trough!)
         (only-in "structs.rkt" tensor? wrap-tensor))

(provide collect-at-forward-trough!
         with-no-grad)

(define/checked-out (requires-grad! t [on? #t])
  (->* [tensor?] [boolean?] tensor?)
  (check-ok (tr-tensor-requires-grad!/raw t on?) 'requires-grad!)
  t)

(define/contract-out (requires-grad? t) (-> tensor? boolean?)
  (define-values (rc on?) (tr-tensor-requires-grad/raw t))
  (check-ok rc 'requires-grad?)
  on?)

(define/contract-out (backward! t) (-> tensor? void?)
  (check-ok (tr-tensor-backward/raw t) 'backward!)
  (collect-at-trough!))

;; With gradients off nothing outlives a forward pass but its result, so the
;; return of an outermost layer call is a trough like the end of backward!.
;; With them on it is the peak: the graph holds every intermediate.
(define (collect-at-forward-trough!)
  (define-values (rc on?) (tr-is-grad-enabled/raw))
  (when (and (zero? rc) (not on?))
    (collect-at-trough! #:young? #t)))

(define/contract-out (grad t) (-> tensor? tensor?)
  (wrap-tensor (check-handle 'grad (tr-tensor-grad/raw t))))

(define/contract-out (has-grad? t) (-> tensor? boolean?)
  (define-values (rc on?) (tr-tensor-has-grad/raw t))
  (check-ok rc 'has-grad?)
  on?)

(define/contract-out (maybe-grad t) (-> tensor? (or/c tensor? #f)) ;; noqa
  (and (has-grad? t) (grad t)))

(define/contract-out (detach t) (-> tensor? tensor?)
  (wrap-tensor (check-handle 'detach (tr-tensor-detach/raw t))))

(define/contract-out (grad-enabled?) (-> boolean?)
  (define-values (rc on?) (tr-is-grad-enabled/raw))
  (check-ok rc 'grad-enabled?)
  on?)

(define (set-grad-enabled! on?)
  (check-ok (tr-set-grad-enabled/raw on?) 'set-grad-enabled!))

(define/contract-out (call-with-no-grad thunk) (-> (-> any) any)
  (define was? (grad-enabled?))
  (dynamic-wind (lambda () (set-grad-enabled! #f))
                thunk
                (lambda () (set-grad-enabled! was?))))

(define-syntax-parse-rule (with-no-grad body:expr ...+)
  (call-with-no-grad (lambda () body ...)))

(define/contract-out (sub! t other [alpha 1.0])
  (->* [tensor? tensor?] [real?] void?)
  (check-ok (tr-tensor-sub!/raw t other (exact->inexact alpha)) 'sub!)
  (void))

(define/contract-out (zero! t) (-> tensor? void?)
  (check-ok (tr-tensor-zero!/raw t) 'zero!)
  (void))

(define/contract-out (mul! t value) (-> tensor? real? void?)
  (check-ok (tr-tensor-mul!/raw t (exact->inexact value)) 'mul!)
  (void))

(define/contract-out (copy! t source) (-> tensor? tensor? void?) ;; noqa
  (void (g:copy! t source #f)))

(define/contract-out (addcmul! t a b [value 1.0]) ;; noqa
  (->* [tensor? tensor? tensor?] [real?] void?)
  (void (g:addcmul! t a b (exact->inexact value))))

(define/contract-out (addcdiv! t a b [value 1.0]) ;; noqa
  (->* [tensor? tensor? tensor?] [real?] void?)
  (void (g:addcdiv! t a b (exact->inexact value))))

(define/contract-out (lerp! t end weight) (-> tensor? tensor? tensor? void?) ;; noqa
  (void (g:lerp-tensor! t end weight)))

(define/contract-out (zero-grad! t) (-> tensor? void?) ;; noqa
  (when (has-grad? t)
    (zero! (grad t))))

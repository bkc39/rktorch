#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c >/c and/c any/c contract-out or/c)
         (only-in "../foreign.rkt" device? tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optim.rkt" optimizer-root optimizer?))

(provide (contract-out [grad-scaler? (-> any/c boolean?)]))

(struct stage (state found-infs))

(struct grad-scaler (target active? growth backoff interval
                            [factor #:mutable] [tracker #:mutable] [stages #:mutable])
  #:constructor-name make-grad-scaler ;; noqa
  #:name grad-scaler-struct) ;; noqa

(define (scaled? s)
  (or (not (grad-scaler-active? s)) (and (grad-scaler-factor s) #t)))

(define (optimizer-stage s opt)
  (define st (hash-ref (grad-scaler-stages s) (optimizer-root opt) #f))
  (if st (stage-state st) 'ready))

(define ((not-at state) s opt)
  (or (not (grad-scaler-active? s)) (not (eq? (optimizer-stage s opt) state))))

(define not-unscaled? (not-at 'unscaled))
(define not-stepped? (not-at 'stepped))

(define (inf-checks-recorded? s)
  (or (not (grad-scaler-active? s))
      (for/or ([st (in-hash-values (grad-scaler-stages s))])
        (pair? (stage-found-infs st)))))

(define/contract-out (grad-scaler #:device [device 'cuda] #:init-scale [init-scale 65536.0]
                                  #:growth-factor [growth-factor 2.0]
                                  #:backoff-factor [backoff-factor 0.5]
                                  #:growth-interval [growth-interval 2000]
                                  #:enabled? [enabled? #t])
  (->* ()
       (#:device (or/c 'cpu 'cuda 'mps device?) #:init-scale (>/c 0) #:growth-factor (>/c 1)
        #:backoff-factor (and/c (>/c 0) (</c 1)) #:growth-interval exact-positive-integer?
        #:enabled? boolean?)
       grad-scaler?)
  (error 'unimplemented))

(define/contract-out (grad-scaler-scale-loss s loss)
  (-> grad-scaler? tensor? tensor?)
  (error 'unimplemented))

(define/contract-out (grad-scaler-unscale! s opt)
  (->i ([s grad-scaler?] [opt optimizer?])
       #:pre/name (s) "no loss has been scaled yet"
       (scaled? s)
       #:pre/name (s opt)
       "grad-scaler-unscale! was already called on this optimizer since the last update"
       (not-unscaled? s opt)
       #:pre/name (s opt) "grad-scaler-unscale! comes before grad-scaler-step!"
       (not-stepped? s opt)
       [_ void?])
  (error 'unimplemented))

(define/contract-out (grad-scaler-step! s opt)
  (->i ([s grad-scaler?] [opt optimizer?])
       #:pre/name (s) "no loss has been scaled yet"
       (scaled? s)
       #:pre/name (s opt)
       "grad-scaler-step! was already called on this optimizer since the last update"
       (not-stepped? s opt)
       [_ void?])
  (error 'unimplemented))

(define/contract-out (grad-scaler-update! s [new-scale #f])
  (->i ([s grad-scaler?]) ([new-scale (or/c #f (>/c 0))])
       #:pre/name (s new-scale) "no inf checks were recorded before the update"
       (inf-checks-recorded? s)
       [_ void?])
  (error 'unimplemented))

(define/contract-out (grad-scaler-scale s)
  (-> grad-scaler? real?)
  (error 'unimplemented))

(define/contract-out (grad-scaler-enabled? s)
  (-> grad-scaler? boolean?)
  (error 'unimplemented))

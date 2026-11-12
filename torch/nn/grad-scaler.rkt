#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c >/c and/c any/c contract-out or/c)
         (only-in racket/match match match*)
         (only-in "../foreign.rkt"
                  * + copy! cuda-available? device device-type device? full item tensor? to)
         (prefix-in g: (only-in "../generated.rkt"
                                addcdiv! amp-foreach-non-finite-check-and-unscale!
                                amp-update-scale! fill-scalar! sum-dim-intlist))
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optim.rkt" optimizer-root optimizer?)
         (only-in (submod "optim.rkt" private)
                  fold-gradient-buckets optimizer-fused? scaled-step!))

(provide (contract-out [grad-scaler? (-> any/c boolean?)]))

(module+ private
  (provide grad-scaler-armed?
           grad-scaler-checked?
           grad-scaler-forget!))

(define-logger rktorch-amp)

(struct stage (state found-infs))

(struct grad-scaler (active? init growth backoff interval
                             [factor #:mutable] [tracker #:mutable] [home #:mutable]
                             [unit #:mutable] [inverse #:mutable] [spares #:mutable]
                             [stages #:mutable])
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

(define (grad-scaler-armed? s)
  (and (grad-scaler-active? s) (grad-scaler-factor s) #t))

(define (grad-scaler-checked? s)
  (and (grad-scaler-active? s) (inf-checks-recorded? s)))

(define (grad-scaler-forget! s opt)
  (set-grad-scaler-stages! s (hash-remove (grad-scaler-stages s) (optimizer-root opt))))

(define (cuda-target? target)
  (eq? 'cuda (if (device? target) (device-type target) target)))

(define (usable? target enabled?)
  (cond
    [(not enabled?) #f]
    [(and (cuda-target? target) (not (cuda-available?)))
     (log-rktorch-amp-warning "grad-scaler is enabled, but CUDA is not available; disabling")
     #f]
    [else #t]))

(define/contract-out (grad-scaler #:device [target 'cuda] #:init-scale [init-scale 65536.0]
                                  #:growth-factor [growth-factor 2.0]
                                  #:backoff-factor [backoff-factor 0.5]
                                  #:growth-interval [growth-interval 2000]
                                  #:enabled? [enabled? #t])
  (->* ()
       (#:device (or/c 'cpu 'cuda 'mps device?) #:init-scale (>/c 0) #:growth-factor (>/c 1)
        #:backoff-factor (and/c (>/c 0) (</c 1)) #:growth-interval exact-positive-integer?
        #:enabled? boolean?)
       grad-scaler?)
  (make-grad-scaler (usable? target enabled?) (exact->inexact init-scale)
                    (exact->inexact growth-factor) (exact->inexact backoff-factor)
                    growth-interval #f #f #f #f #f '() #hasheq()))

(define (lazy-init! s dev)
  (set-grad-scaler-factor! s (full (grad-scaler-init s) #:device dev #:dtype 'float32))
  (set-grad-scaler-tracker! s (g:sum-dim-intlist (full 0.0 #:device dev) #f #f 'int32))
  (set-grad-scaler-unit! s (full 1.0 #:device dev #:dtype 'float32))
  (set-grad-scaler-inverse! s (full 0.0 #:device dev #:dtype 'float32))
  (set-grad-scaler-home! s dev))

(define/contract-out (grad-scaler-scale-loss s loss)
  (-> grad-scaler? tensor? tensor?)
  (cond
    [(not (grad-scaler-active? s)) loss]
    [else
     (unless (grad-scaler-factor s)
       (lazy-init! s (device loss)))
     (* loss (grad-scaler-factor s))]))

(define (set-stage! s opt state found-infs)
  (set-grad-scaler-stages! s (hash-set (grad-scaler-stages s) (optimizer-root opt)
                                       (stage state found-infs))))

(define (stage-infs s opt)
  (stage-found-infs (hash-ref (grad-scaler-stages s) (optimizer-root opt))))

(define (fresh-found! s dev)
  (define spares (grad-scaler-spares s))
  (match (assoc dev spares)
    [#f (full 0.0 #:device dev #:dtype 'float32)]
    [(and spare (cons _ found))
     (set-grad-scaler-spares! s (remq spare spares))
     (g:fill-scalar! found 0.0)
     found]))

(define (on-device t home dev)
  (if (equal? dev home) t (to t dev)))

(define (check-and-unscale! s opt inv)
  (define home (grad-scaler-home s))
  (define (check! found-infs dev grads)
    (define held (assoc dev found-infs))
    (define found (if held (cdr held) (fresh-found! s dev)))
    (g:amp-foreach-non-finite-check-and-unscale! grads found (on-device inv home dev))
    (if held found-infs (cons (cons dev found) found-infs)))
  (reverse (fold-gradient-buckets opt '() check!)))

(define (inverse-scale s)
  (define inverse (grad-scaler-inverse s))
  (g:fill-scalar! inverse 0.0)
  (g:addcdiv! inverse (grad-scaler-unit s) (grad-scaler-factor s) 1.0)
  inverse)

(define (unscale! s opt)
  (set-stage! s opt 'unscaled (check-and-unscale! s opt (inverse-scale s))))

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
  (when (grad-scaler-active? s)
    (unscale! s opt)))

(define (combined-inf home found-infs)
  (match (for/list ([entry (in-list found-infs)])
           (on-device (cdr entry) (car entry) home))
    ['() #f]
    [(cons first rest) (for/fold ([acc first]) ([f (in-list rest)]) (+ acc f))]))

(define (fused-step! s opt)
  (define unscaled? (eq? (optimizer-stage s opt) 'unscaled))
  (define found-infs
    (if unscaled? (stage-infs s opt) (check-and-unscale! s opt (grad-scaler-unit s))))
  (scaled-step! opt (if unscaled? #f (grad-scaler-factor s))
                (combined-inf (grad-scaler-home s) found-infs))
  (set-stage! s opt 'stepped found-infs))

(define (checked-step! s opt)
  (when (eq? (optimizer-stage s opt) 'ready)
    (unscale! s opt))
  (define found-infs (stage-infs s opt))
  (when (zero? (for/sum ([entry (in-list found-infs)]) (item (cdr entry))))
    (scaled-step! opt))
  (set-stage! s opt 'stepped found-infs))

(define/contract-out (grad-scaler-step! s opt)
  (->i ([s grad-scaler?] [opt optimizer?])
       #:pre/name (s) "no loss has been scaled yet"
       (scaled? s)
       #:pre/name (s opt)
       "grad-scaler-step! was already called on this optimizer since the last update"
       (not-stepped? s opt)
       [_ void?])
  (cond
    [(not (grad-scaler-active? s)) (scaled-step! opt)]
    [(optimizer-fused? opt) (fused-step! s opt)]
    [else (checked-step! s opt)]))

(define (all-infs s)
  (for*/list ([st (in-hash-values (grad-scaler-stages s))]
              [entry (in-list (stage-found-infs st))])
    entry))

(define/contract-out (grad-scaler-update! s [new-scale #f])
  (->i ([s grad-scaler?]) ([new-scale (or/c #f (>/c 0))])
       #:pre/name (s new-scale) "no inf checks were recorded before the update"
       (inf-checks-recorded? s)
       [_ void?])
  (define factor (grad-scaler-factor s))
  (cond
    [(not (grad-scaler-active? s)) (void)]
    [new-scale
     (copy! factor (full (exact->inexact new-scale) #:device (grad-scaler-home s)
                         #:dtype 'float32))]
    [else
     (g:amp-update-scale! factor (grad-scaler-tracker s)
                          (combined-inf (grad-scaler-home s) (all-infs s))
                          (grad-scaler-growth s) (grad-scaler-backoff s)
                          (grad-scaler-interval s))])
  (set-grad-scaler-spares! s (append (all-infs s) (grad-scaler-spares s)))
  (set-grad-scaler-stages! s #hasheq())
  (void))

(define/contract-out (grad-scaler-scale s)
  (-> grad-scaler? real?)
  (match* ((grad-scaler-active? s) (grad-scaler-factor s))
    [(#f _) 1.0]
    [(_ #f) (grad-scaler-init s)]
    [(_ factor) (item factor)]))

(define/contract-out (grad-scaler-enabled? s)
  (-> grad-scaler? boolean?)
  (grad-scaler-active? s))

#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c >/c and/c any/c contract-out or/c unsupplied-arg?)
         (only-in racket/match match match* match-define struct*)
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
           grad-scaler-forget!
           host-reads))

(define-logger rktorch-amp)

(struct inf-check (device found))

(struct stage (state found-infs))

(struct device-state (factor tracker unit inverse home
                             [spares #:mutable] [stages #:mutable]))

(struct grad-scaler (active? init growth backoff interval [state #:auto #:mutable])
  #:auto-value #f
  #:constructor-name make-grad-scaler ;; noqa
  #:name grad-scaler-struct) ;; noqa

(define reads (box 0))

(define (host-reads) (unbox reads))

(define (scaled? s)
  (or (not (grad-scaler-active? s)) (and (grad-scaler-state s) #t)))

(define (stages-of s)
  (match (grad-scaler-state s)
    [#f #hasheq()]
    [st (device-state-stages st)]))

(define (optimizer-stage s opt)
  (match (hash-ref (stages-of s) (optimizer-root opt) #f)
    [#f 'ready]
    [(struct* stage ([state state])) state]))

(define ((not-at state) s opt)
  (or (not (grad-scaler-active? s)) (not (eq? (optimizer-stage s opt) state))))

(define not-unscaled? (not-at 'unscaled))
(define not-stepped? (not-at 'stepped))

(define (inf-checks-recorded? s)
  (or (not (grad-scaler-active? s))
      (for/or ([st (in-hash-values (stages-of s))])
        (pair? (stage-found-infs st)))))

(define (grad-scaler-armed? s)
  (and (grad-scaler-active? s) (grad-scaler-state s) #t))

(define (grad-scaler-checked? s)
  (and (grad-scaler-active? s) (inf-checks-recorded? s)))

(define (grad-scaler-forget! s opt)
  (define st (grad-scaler-state s))
  (set-device-state-stages! st (hash-remove (device-state-stages st) (optimizer-root opt))))

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
                    growth-interval))

(define (fresh-state s dev)
  (device-state (full (grad-scaler-init s) #:device dev #:dtype 'float32)
                (g:sum-dim-intlist (full 0.0 #:device dev) #f #f 'int32)
                (full 1.0 #:device dev #:dtype 'float32)
                (full 0.0 #:device dev #:dtype 'float32)
                dev '() #hasheq()))

(define/contract-out (grad-scaler-scale-loss s loss)
  (-> grad-scaler? tensor? tensor?)
  (cond
    [(not (grad-scaler-active? s)) loss]
    [else
     (unless (grad-scaler-state s)
       (set-grad-scaler-state! s (fresh-state s (device loss))))
     (* loss (device-state-factor (grad-scaler-state s)))]))

(define (set-stage! st opt state found-infs)
  (set-device-state-stages! st (hash-set (device-state-stages st) (optimizer-root opt)
                                         (stage state found-infs))))

(define (stage-infs st opt)
  (stage-found-infs (hash-ref (device-state-stages st) (optimizer-root opt))))

(define (fresh-found! st dev)
  (define spares (device-state-spares st))
  (match (findf (lambda (c) (equal? (inf-check-device c) dev)) spares)
    [#f (full 0.0 #:device dev #:dtype 'float32)]
    [(and spare (struct* inf-check ([found found])))
     (set-device-state-spares! st (remq spare spares))
     (g:fill-scalar! found 0.0)
     found]))

(define (on-device t home dev)
  (if (equal? dev home) t (to t dev)))

(define (check-and-unscale! st opt inv #:who [who #f])
  (define home (device-state-home st))
  (define (check! found-infs dev dt grads)
    (when (and who (eq? dt 'float16))
      (raise-arguments-error who "attempting to unscale FP16 gradients" "optimizer" opt))
    (define held (findf (lambda (c) (equal? (inf-check-device c) dev)) found-infs))
    (define found (if held (inf-check-found held) (fresh-found! st dev)))
    (g:amp-foreach-non-finite-check-and-unscale! grads found (on-device inv home dev))
    (if held found-infs (cons (inf-check dev found) found-infs)))
  (reverse (fold-gradient-buckets opt '() check!)))

(define (inverse-scale st)
  (define inverse (device-state-inverse st))
  (g:fill-scalar! inverse 0.0)
  (g:addcdiv! inverse (device-state-unit st) (device-state-factor st) 1.0)
  inverse)

(define (unscale! st opt who)
  (set-stage! st opt 'unscaled (check-and-unscale! st opt (inverse-scale st) #:who who)))

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
    (unscale! (grad-scaler-state s) opt 'grad-scaler-unscale!)))

(define (combined-inf home found-infs)
  (match (for/list ([c (in-list found-infs)])
           (match-define (struct* inf-check ([device dev] [found found])) c)
           (on-device found dev home))
    ['() #f]
    [(cons first rest) (for/fold ([acc first]) ([f (in-list rest)]) (+ acc f))]))

(define (fused-step! st opt unscaled?)
  (define found-infs
    (if unscaled? (stage-infs st opt) (check-and-unscale! st opt (device-state-unit st))))
  (scaled-step! opt (if unscaled? #f (device-state-factor st))
                (combined-inf (device-state-home st) found-infs))
  (set-stage! st opt 'stepped found-infs))

(define (inf-count! found-infs)
  (set-box! reads (add1 (unbox reads)))
  (for/sum ([c (in-list found-infs)]) (item (inf-check-found c))))

(define (checked-step! st opt ready?)
  (when ready?
    (unscale! st opt 'grad-scaler-step!))
  (define found-infs (stage-infs st opt))
  (when (zero? (inf-count! found-infs))
    (scaled-step! opt))
  (set-stage! st opt 'stepped found-infs))

(define/contract-out (grad-scaler-step! s opt)
  (->i ([s grad-scaler?] [opt optimizer?])
       #:pre/name (s) "no loss has been scaled yet"
       (scaled? s)
       #:pre/name (s opt)
       "grad-scaler-step! was already called on this optimizer since the last update"
       (not-stepped? s opt)
       [_ void?])
  (define st (grad-scaler-state s))
  (define ready? (eq? (optimizer-stage s opt) 'ready))
  (cond
    [(not (grad-scaler-active? s)) (scaled-step! opt)]
    [(optimizer-fused? opt) (fused-step! st opt (not ready?))]
    [else (checked-step! st opt ready?)]))

(define (all-infs st)
  (for*/list ([sg (in-hash-values (device-state-stages st))]
              [c (in-list (stage-found-infs sg))])
    c))

(define (settle! s st new-scale)
  (match-define (struct* device-state ([factor factor] [tracker tracker] [home home])) st)
  (match new-scale
    [#f
     (g:amp-update-scale! factor tracker (combined-inf home (all-infs st))
                          (grad-scaler-growth s) (grad-scaler-backoff s)
                          (grad-scaler-interval s))]
    [scale (copy! factor (full (exact->inexact scale) #:device home #:dtype 'float32))])
  (set-device-state-spares! st (append (all-infs st) (device-state-spares st)))
  (set-device-state-stages! st #hasheq()))

(define (given-scale? v)
  (and v (not (unsupplied-arg? v))))

(define/contract-out (grad-scaler-update! s [new-scale #f])
  (->i ([s grad-scaler?]) ([new-scale (or/c #f (>/c 0))])
       #:pre/name (s) "no loss has been scaled yet"
       (scaled? s)
       #:pre/name (s new-scale) "no inf checks were recorded before the update"
       (or (given-scale? new-scale) (inf-checks-recorded? s))
       [_ void?])
  (when (grad-scaler-active? s)
    (settle! s (grad-scaler-state s) new-scale)))

(define/contract-out (grad-scaler-scale s)
  (-> grad-scaler? real?)
  (match* ((grad-scaler-active? s) (grad-scaler-state s))
    [(#f _) 1.0]
    [(_ #f) (grad-scaler-init s)]
    [(_ st) (item (device-state-factor st))]))

(define/contract-out (grad-scaler-enabled? s)
  (-> grad-scaler? boolean?)
  (grad-scaler-active? s))

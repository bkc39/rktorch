#lang racket/base

;; DIAGNOSTIC ONLY (#439): per-site counters and wall time, split by whether
;; the caller is on the place's main OS thread. Off unless RKTORCH_DIAG439
;; is set. Not for merge.

(require (for-syntax racket/base)
         (only-in ffi/unsafe/vm vm-eval))

(provide diag-enabled?
         diag-no-phantom?
         diag-finalizer-pool
         diag-sites
         diag-on-main?
         diag-add!
         diag-count!
         diag-timed
         diag-snapshot
         diag-reset!)

(define diag-enabled? (and (getenv "RKTORCH_DIAG439") #t))
(define diag-no-phantom? (and (getenv "RKTORCH_DIAG439_NOPHANTOM") #t))
;; #f: ffi/unsafe register-finalizer (a green thread on main's OS thread);
;; n: n parallel 'own threads executing a private late will executor
(define diag-finalizer-pool
  (let ([v (getenv "RKTORCH_DIAG439_FINALIZERS")]) (and v (string->number v))))

(define diag-sites
  '(finalizer finalizer/worker-made finalizer/main-made release-native
    register account cup ledger-acquire ledger-wait
    trough trough/collect-and-wait trough/minor drain drain/timeout
    pressure-collect yield-sleep collect-and-wait canary-wait))

(define os-thread-id (vm-eval 'get-thread-id))
(define main-os-thread
  (let ([id (box #f)])
    (thread-wait (thread (lambda () (set-box! id (os-thread-id)))))
    (unbox id)))

(define (diag-on-main?)
  (eqv? (os-thread-id) main-os-thread))

;; site -> vector of 4 boxes: main count, main ns, other count, other ns
(define table
  (for/hasheq ([s (in-list diag-sites)])
    (values s (vector (box 0) (box 0) (box 0) (box 0)))))

(define (bump! b n)
  (let retry ()
    (define old (unbox b))
    (unless (box-cas! b old (+ old n)) (retry))))

(define (diag-add! site ns)
  (define v (hash-ref table site))
  (define off (if (diag-on-main?) 0 2))
  (bump! (vector-ref v off) 1)
  (bump! (vector-ref v (add1 off)) ns))

(define (diag-count! site)
  (when diag-enabled? (diag-add! site 0)))

(define (now-ns)
  (inexact->exact (floor (* 1e6 (current-inexact-monotonic-milliseconds)))))

(define-syntax (diag-timed stx)
  (syntax-case stx ()
    [(_ site body ...)
     #'(if diag-enabled?
           (let ([t0 (now-ns)])
             (begin0 (let () body ...)
                     (diag-add! site (- (now-ns) t0))))
           (let () body ...))]))

(define (diag-snapshot)
  (for/hasheq ([(s v) (in-hash table)])
    (values s (for/list ([b (in-vector v)]) (unbox b)))))

(define (diag-reset!)
  (for* ([v (in-hash-values table)] [b (in-vector v)])
    (set-box! b 0)))

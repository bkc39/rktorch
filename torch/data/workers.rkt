#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/custodian
                  make-custodian-at-root register-custodian-shutdown
                  unregister-custodian-shutdown)
         (only-in racket/contract/base -> any/c or/c)
         (only-in racket/match match)
         (only-in "../foreign.rkt"
                  current-generator device device-type make-generator tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "dataset.rkt" collate/c dataset-batch dataset? indices/c))

(provide (struct-out built)
         (struct-out raised))

(module+ private
  (provide (struct-out pool-guard) worker-custodian))

(struct built (values-list))
(struct raised (value))

(define batch-result/c (or/c built? raised?))

;; Workers never reach the pool, so its finalizer can stop them.
(struct worker-pool (threads results ready stop))

;; Racket 9.3's custodian-shutdown-all holds the custodian lock while it waits
;; for a running parallel thread that needs the lock to stop. Workers live under
;; a custodian of their own, and the traversal's custodian holds a guard whose
;; shutdown callback, run under that lock, only wakes the reaper.
(struct pool-guard (threads kill))

(define worker-custodian (make-custodian-at-root))

(define (on-host? v)
  (or (not (tensor? v)) (eq? (device-type (device v)) 'cpu)))

(define (build ds collate r indices)
  (with-handlers ([(lambda (v) (not (exn:break? v))) raised])
    (define values-list
      (call-with-values (lambda () (dataset-batch ds indices collate)) list))
    (cond
      [(andmap on-host? values-list) (built values-list)]
      [else
       (raise-arguments-error
        'dataloader
        "workers build CPU tensors, but a batch holds one on an accelerator"
        "batch" r)])))

(define (worker-prng seed)
  (define prng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator prng])
    (random-seed (bitwise-and seed #x7FFFFFFF)))
  prng)

(define (serve ds collate results ready stop)
  (define stopped (semaphore-peek-evt stop))
  (define request (handle-evt (thread-receive-evt) (lambda (_) (thread-receive))))
  (let loop ()
    (match (and (not (sync/timeout 0 stopped)) (sync stopped request))
      [(cons r indices)
       (hash-set! results r (build ds collate r indices))
       (semaphore-post ready)
       (loop)]
      [_ (void)])))

(define (reap guard registration stop)
  (define kill (pool-guard-kill guard))
  (define woken (sync (semaphore-peek-evt stop) kill))
  (unregister-custodian-shutdown guard registration)
  (for ([t (in-vector (pool-guard-threads guard))]
        #:when (eq? woken kill))
    (kill-thread t)))

(define/contract-out (start-workers ds collate n base-seed) ;; noqa
  (-> dataset? collate/c exact-positive-integer? exact-nonnegative-integer?
      worker-pool?)
  (define results (make-hasheqv))
  (define ready (make-semaphore 0))
  (define stop (make-semaphore 0))
  (define caller (current-custodian))
  (define threads
    (parameterize ([current-custodian worker-custodian])
      (for/vector #:length n ([k (in-range n)])
        (define seed (+ base-seed k))
        (thread (lambda ()
                  (parameterize ([current-custodian caller]
                                 [current-generator (make-generator seed)]
                                 [current-pseudo-random-generator (worker-prng seed)])
                    (serve ds collate results ready stop)))
                #:pool 'own))))
  (define guard (pool-guard threads (make-semaphore 0)))
  (define registration
    (register-custodian-shutdown guard
                                 (lambda (g) (semaphore-post (pool-guard-kill g)))
                                 caller))
  (cond
    [registration
     (parameterize ([current-custodian worker-custodian])
       (thread (lambda () (reap guard registration stop))))]
    [else
     (semaphore-post stop)
     (raise-arguments-error 'dataloader "the custodian has been shut down"
                            "custodian" caller)])
  (define pool (worker-pool threads results ready stop))
  (register-finalizer pool (lambda (_pool) (semaphore-post stop)))
  pool)

(define (worker-of pool r)
  (define threads (worker-pool-threads pool))
  (modulo r (vector-length threads)))

(define/contract-out (hand-off! pool r indices) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? indices/c void?)
  (void (thread-send (vector-ref (worker-pool-threads pool) (worker-of pool r))
                     (cons r indices)
                     #f)))

(define/contract-out (await-batch pool r breakable?) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? boolean? batch-result/c)
  (define results (worker-pool-results pool))
  (define k (worker-of pool r))
  (define dead (thread-dead-evt (vector-ref (worker-pool-threads pool) k)))
  (define wait (if breakable? sync/enable-break sync))
  (let loop ([exited? #f])
    (define result (hash-ref results r #f))
    (cond
      [result
       (hash-remove! results r)
       result]
      [exited?
       (raised (make-exn:fail
                (format "dataloader: worker ~a exited before delivering batch ~a" k r)
                (current-continuation-marks)))]
      [else (loop (eq? (wait (worker-pool-ready pool) dead) dead))])))

(define/contract-out (stop-workers! pool) (-> worker-pool? void?) ;; noqa
  (semaphore-post (worker-pool-stop pool)))

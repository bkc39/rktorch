#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/custodian
                  make-custodian-at-root register-custodian-shutdown
                  unregister-custodian-shutdown)
         (only-in racket/contract/base -> any/c or/c)
         (only-in racket/match match match-define)
         (only-in "../foreign.rkt"
                  current-generator device device-type make-generator tensor?)
         (only-in "../foreign/raw/race-points.rkt" race-point)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "dataset.rkt" collate/c dataset-batch dataset? indices/c)
         (only-in "results-queue.rkt"
                  results-queue results-queue-put! results-queue-take!))

(provide (struct-out built)
         (struct-out raised))

(module+ private
  (provide (struct-out pool-guard) worker-custodian))

(struct built (values-list))
(struct raised (value))

(define batch-result/c (or/c built? raised?))

;; Workers never reach the pool, so its finalizer can stop them.
(struct worker-pool (guard results))

;; The guard's shutdown callback runs under the custodian lock, so it only posts.
(struct pool-guard ([threads #:mutable] stop kill))

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

(define (serve ds collate results stop)
  (define stopped (semaphore-peek-evt stop))
  (define request (handle-evt (thread-receive-evt) (lambda (_) (thread-receive))))
  (let loop ()
    (match (and (not (sync/timeout 0 stopped)) (sync stopped request))
      [(cons r indices)
       (results-queue-put! results r (build ds collate r indices))
       (loop)]
      [_ (void)])))

(define (reap guard registration)
  (match-define (pool-guard threads _ kill) guard)
  (define killed?
    (for/or ([t (in-vector threads)])
      (eq? (sync kill (thread-dead-evt t)) kill)))
  (for ([t (in-vector threads)] #:when killed?)
    (kill-thread t))
  (unregister-custodian-shutdown guard registration))

(define (guard-with! caller guard)
  (or (register-custodian-shutdown guard
                                   (lambda (g)
                                     (semaphore-post (pool-guard-stop g))
                                     (semaphore-post (pool-guard-kill g)))
                                   caller)
      (raise-arguments-error 'dataloader "the custodian has been shut down"
                             "custodian" caller)))

(define (start-reaper! guard registration)
  (parameterize ([current-custodian worker-custodian])
    (void (thread (lambda () (reap guard registration))))))

(define (start-threads ds collate n base-seed results stop breaks)
  (define caller (current-custodian))
  (parameterize ([current-custodian worker-custodian])
    (for/vector #:length n ([k (in-range n)])
      (define seed (+ base-seed k))
      (thread (lambda ()
                (call-with-break-parameterization
                 breaks
                 (lambda ()
                   (parameterize ([current-custodian caller]
                                  [current-generator (make-generator seed)]
                                  [current-pseudo-random-generator (worker-prng seed)])
                     (serve ds collate results stop)))))
              #:pool 'own))))

(define/contract-out (start-workers ds collate n base-seed) ;; noqa
  (-> dataset? collate/c exact-positive-integer? exact-nonnegative-integer?
      worker-pool?)
  (define breaks (current-break-parameterization))
  (parameterize-break #f
    (define results (results-queue))
    (define stop (make-semaphore 0))
    ;; Both ways to stop exist before the first worker does.
    (define guard (pool-guard (vector) stop (make-semaphore 0)))
    (define pool (worker-pool guard results))
    (register-finalizer pool (lambda (_pool) (semaphore-post stop)))
    (define registration (guard-with! (current-custodian) guard))
    (define threads (start-threads ds collate n base-seed results stop breaks))
    (set-pool-guard-threads! guard threads)
    (race-point workers-started threads)
    (start-reaper! guard registration)
    pool))

(define (worker-pool-threads pool)
  (pool-guard-threads (worker-pool-guard pool)))

(define (worker-of pool r)
  (modulo r (vector-length (worker-pool-threads pool))))

(define/contract-out (request-batch! pool r indices) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? indices/c void?)
  (void (thread-send (vector-ref (worker-pool-threads pool) (worker-of pool r))
                     (cons r indices)
                     #f)))

(define/contract-out (receive-batch pool r breakable?) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? boolean? batch-result/c)
  (define k (worker-of pool r))
  (define dead (thread-dead-evt (vector-ref (worker-pool-threads pool) k)))
  (or (results-queue-take! (worker-pool-results pool) r dead breakable?)
      (raised (make-exn:fail
               (format "dataloader: worker ~a exited before delivering batch ~a" k r)
               (current-continuation-marks)))))

(define/contract-out (stop-workers! pool) (-> worker-pool? void?) ;; noqa
  (semaphore-post (pool-guard-stop (worker-pool-guard pool))))

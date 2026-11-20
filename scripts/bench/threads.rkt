#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in racket/match match match* match-define)
         (only-in racket/promise delay force)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in torch + @ backward! randn relu sum tensor)
         (only-in torch/nn Linear Sequential cross-entropy parameters sgd step! zero-grads!)
         (only-in "harness.rkt" bench-case clock-ms))

(provide call-with-workers
         thread-counts
         main-states
         mix-names
         ops-per-iteration
         peak-of
         scaling-run
         scaling-cases
         training-step)

(module+ private
  (provide collect-with-allocating-threads!
           collector-sweepers
           os-thread-count
           peak-of
           run-worker!
           settled-thread-count
           task-dir
           with-peak-sampler))

(define thread-counts '(1 2 4 8))
(define main-states '(idle train))
(define ops-per-iteration 4)
(define task-dir "/proc/self/task")
(define sample-interval 0.1)

(define (small-mix)
  (define a8 (randn 8 8))
  (define b8 (randn 8 8))
  (define a64 (randn 64 64))
  (define b64 (randn 64 64))
  (lambda ()
    (+ a8 b8)
    (@ a64 b64)
    (relu a64)
    (sum b64)))

(define (grain-mix)
  (define inputs (delay (list (randn 256 256) (randn 256 256))))
  (lambda ()
    (match-define (list a b) (force inputs))
    (@ a b)
    (+ a b)))

(struct op-mix (name make ops))

(define mixes
  (list (op-mix 'small small-mix ops-per-iteration)
        (op-mix 'grain grain-mix 2)))

(define mix-names (map op-mix-name mixes))

(define (mix-named name)
  (findf (lambda (m) (eq? (op-mix-name m) name)) mixes))

(define (os-thread-count [tasks task-dir])
  (and (directory-exists? tasks) (length (directory-list tasks))))

(define (settled-thread-count tasks #:interval [interval sample-interval] #:within [limit 2.0])
  (define deadline (+ (clock-ms) (* 1000.0 limit)))
  (let loop ([last (os-thread-count tasks)])
    (sleep interval)
    (define now (os-thread-count tasks))
    (if (or (equal? now last) (>= (clock-ms) deadline)) now (loop now))))

(define (peak-of counts)
  (and (andmap values counts) (apply max counts)))

;; Chez's collector starts a sweeper OS thread, up to 16, when more threads
;; wait on a collection than it has sweepers, and keeps it for the life of the
;; process; priming past a run's threads keeps the run from starting any.
(define sweeper-spinners 32)

(define (collect-with-allocating-threads! k)
  (define started (make-semaphore 0))
  (define stop? (box #f))
  (define spinners
    (for/list ([_ (in-range k)])
      (thread (lambda ()
                (define sink (box #f))
                (semaphore-post started)
                (let loop ()
                  (unless (unbox stop?)
                    (set-box! sink (make-vector 16))
                    (loop))))
              #:pool 'own)))
  (dynamic-wind void
                (lambda ()
                  (for ([_ (in-range k)]) (semaphore-wait started))
                  (collect-garbage 'major))
                (lambda ()
                  (set-box! stop? #t)
                  (for-each thread-wait spinners))))

(define (start-collector-sweepers! tasks)
  (let loop ([before (settled-thread-count tasks)] [rounds 4])
    (collect-with-allocating-threads! sweeper-spinners)
    (define after (settled-thread-count tasks))
    (if (or (equal? after before) (zero? rounds)) after (loop after (sub1 rounds)))))

(define collector-sweepers (delay (start-collector-sweepers! task-dir)))

(define (call-with-peak-sampler proc #:tasks [tasks task-dir] #:interval [interval sample-interval])
  (define peak (box (os-thread-count tasks)))
  (define (sample!) (set-box! peak (peak-of (list (unbox peak) (os-thread-count tasks)))))
  (define sampler
    (thread (lambda ()
              (let loop ()
                (sleep interval)
                (sample!)
                (loop)))))
  (define result (dynamic-wind void proc (lambda () (kill-thread sampler))))
  (sample!)
  (values result (unbox peak)))

(define-syntax-parse-rule (with-peak-sampler ((~alt (~optional (~seq #:tasks tasks:expr))
                                                    (~optional (~seq #:interval interval:expr)))
                                              ...)
                            body:expr ...+)
  (call-with-peak-sampler (lambda () body ...)
                          (~? (~@ #:tasks tasks))
                          (~? (~@ #:interval interval))))

(define (training-step)
  (define net (Sequential (Linear 784 256) relu (Linear 256 10)))
  (define opt (sgd (parameters net) #:lr 0.01))
  (define xs (randn 64 784))
  (define ys (tensor (for/list ([i (in-range 64)]) (modulo i 10)) #:dtype 'int64))
  (lambda ()
    (zero-grads! opt)
    (backward! (cross-entropy (net xs) ys))
    (step! opt)))

(define (run-main state deadline)
  (match state
    ['idle (sync (alarm-evt deadline #t)) 0]
    ['train
     (define step! (training-step))
     (let loop ([steps 0])
       (cond
         [(>= (clock-ms) deadline) steps]
         [else (step!) (loop (add1 steps))]))]))

(define (run-worker! phase mix count!)
  (let loop ([k 0])
    (match* ((unbox phase) k)
      [('wait _) (loop k)]
      [('abort _) (void)]
      [('stop (? positive?)) (void)]
      [(_ _)
       (mix)
       (count! (add1 k))
       (loop (add1 k))])))

(define (start-workers n phase counts failure make-mix)
  (define mixes (for/list ([_ (in-range n)]) (make-mix)))
  (define custodian (make-custodian-at-root))
  (define (fail! e)
    (let retry () (unless (or (unbox failure) (box-cas! failure #f e)) (retry)))
    (set-box! phase 'stop))
  (values custodian
          (parameterize ([current-custodian custodian])
            (for/list ([mix (in-list mixes)] [i (in-naturals)])
              (thread (lambda ()
                        (with-handlers ([(lambda (_e) #t) fail!])
                          (run-worker! phase mix (lambda (k) (vector-set! counts i k)))))
                      #:pool 'own)))))

(define (reap! custodian workers grace)
  (for/sum ([w (in-list workers)])
    (cond
      [(sync/timeout grace w) 1]
      [else (parameterize ([current-custodian custodian]) (kill-thread w)) 0])))

(define (call-with-workers n make-mix grace proc)
  (define phase (box 'wait))
  (define counts (make-vector n 0))
  (define failure (box #f))
  (define-values (custodian workers) (start-workers n phase counts failure make-mix))
  (define (stop!)
    (set-box! phase (if (eq? (unbox phase) 'wait) 'abort 'stop))
    (reap! custodian workers grace))
  (define result
    (with-handlers ([(lambda (_e) #t) (lambda (e) (stop!) (raise e))])
      (proc (lambda () (set-box! phase 'go))
            (lambda () (for/sum ([k (in-vector counts)]) k)))))
  (define stopped (stop!))
  (match (unbox failure)
    [#f (values result stopped)]
    [e (raise e)]))

(define (scaling-run n state
                     #:seconds seconds
                     #:grace [grace 30]
                     #:mix [make-mix small-mix]
                     #:ops-per-mix [ops-per-mix ops-per-iteration]
                     #:probe [probe void]
                     #:tasks [tasks task-dir])
  (force collector-sweepers)
  (define baseline (settled-thread-count tasks))
  (define-values (run stopped)
    (call-with-workers
     n make-mix grace
     (lambda (go! mixes-done)
       (define-values (timed peak)
         (with-peak-sampler (#:tasks tasks)
           (define probe-start (probe))
           (define t0 (clock-ms))
           (go!)
           (define main-steps (run-main state (+ t0 (* 1000.0 seconds))))
           (define elapsed-s (/ (- (clock-ms) t0) 1000.0))
           (define ops (mixes-done))
           (hasheq 'ops_per_s (/ (* ops-per-mix ops) elapsed-s)
                   'mixes ops
                   'elapsed_s elapsed-s
                   'main_steps_per_s (/ main-steps elapsed-s)
                   'started_ms t0
                   'probe_start probe-start
                   'probe_stop (probe))))
       (hash-set* timed 'peak_os_threads peak 'baseline_os_threads baseline))))
  (hash-set run 'stopped stopped))

(define (scaling-cases #:seconds seconds
                       #:threads [threads thread-counts]
                       #:states [states main-states]
                       #:mix [mix 'small]
                       #:on-run [on-run void])
  (define chosen (mix-named mix))
  (define make-mix (op-mix-make chosen))
  (define ops-per-mix (op-mix-ops chosen))
  (for*/list ([state (in-list states)] [n (in-list threads)])
    (bench-case (string->symbol (format "~a/~a" state n))
                (lambda (_reps)
                  (define run (scaling-run n state #:seconds seconds #:mix make-mix
                                           #:ops-per-mix ops-per-mix))
                  (on-run state n run)
                  (hash-ref run 'ops_per_s))
                #:self-timed? #t)))

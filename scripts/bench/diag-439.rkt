#lang racket/base

;; DIAGNOSTIC ONLY (#439), not for merge. One run: N 'own workers on a mix
;; while main idles or trains; prints one JSON line with main's per-step
;; times, GC by OS thread, the diag.rkt sites, CAS failures per site and
;; /proc schedstat per task.
;;
;; racket scripts/bench/diag-439.rkt N MIX STATE SECONDS
;;   MIX: real | inplace | cons | spin | idle-mix
;;   STATE: train | idle

(require ffi/unsafe
         ffi/unsafe/vm
         json
         racket/flonum
         racket/list
         racket/match
         racket/string
         (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in torch + @ backward! randn relu sum tensor ones zeros)
         (only-in torch/generated add-tensor! addcmul! copy! mul-tensor!)
         (only-in torch/nn Linear Sequential cross-entropy parameters sgd step! zero-grads!)
         (only-in torch/foreign/raw/cas cas-sites)
         (only-in (submod torch/foreign/raw/cas private) site-cas)
         (only-in torch/foreign/raw/diag diag-snapshot diag-reset! diag-enabled?)
         (only-in torch/foreign/raw/memory finalizer-diagnostics))

(define gettid (get-ffi-obj "gettid" #f (_fun -> _int)))
(define os-thread-id (vm-eval 'get-thread-id))
(define main-chez-id
  (let ([id (box #f)])
    (thread-wait (thread (lambda () (set-box! id (os-thread-id)))))
    (unbox id)))

;; --- GC by the OS thread that ran the collect-request handler ---
(define gc-acc
  (vm-eval
   `(let ([acc (make-vector 4 0)]
          [old (collect-request-handler)]
          [main ,main-chez-id])
      (collect-request-handler
       (lambda ()
         (let ([t0 (current-time 'time-monotonic)])
           (old)
           (let* ([t1 (current-time 'time-monotonic)]
                  [ns (+ (* (- (time-second t1) (time-second t0)) 1000000000)
                         (- (time-nanosecond t1) (time-nanosecond t0)))]
                  [off (if (eqv? (get-thread-id) main) 0 2)])
             (vector-set! acc off (+ 1 (vector-ref acc off)))
             (vector-set! acc (+ off 1) (+ ns (vector-ref acc (+ off 1))))))))
      acc)))

(define (gc-acc-snapshot) (for/list ([x (in-vector gc-acc)]) x))

;; --- CAS failures per site ---
(define (bump! b n)
  (let retry () (define o (unbox b)) (unless (box-cas! b o (+ o n)) (retry))))
(define cas-tries (for/hasheq ([s (in-list cas-sites)]) (values s (box 0))))
(define cas-fails (for/hasheq ([s (in-list cas-sites)]) (values s (box 0))))
(when diag-enabled?
  (define prim (unbox site-cas))
  (set-box! site-cas
            (lambda (site target old new)
              (bump! (hash-ref cas-tries site) 1)
              (or (prim site target old new)
                  (begin (bump! (hash-ref cas-fails site) 1) #f)))))
(define (cas-snapshot)
  (for/hasheq ([s (in-list cas-sites)])
    (values s (list (unbox (hash-ref cas-tries s)) (unbox (hash-ref cas-fails s))))))

;; --- schedstat ---
(define (schedstat tid)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (call-with-input-file (format "/proc/self/task/~a/schedstat" tid)
      (lambda (in) (map string->number (string-split (read-line in)))))))
(define (all-tasks)
  (for/list ([p (in-list (directory-list "/proc/self/task"))])
    (string->number (path->string p))))
(define (sched-all)
  (for/hasheqv ([t (in-list (all-tasks))]) (values t (schedstat t))))
(define (loadavg)
  (call-with-input-file "/proc/loadavg" read-line))

;; --- mixes ---
(define (real-mix)
  (define a8 (randn 8 8)) (define b8 (randn 8 8))
  (define a64 (randn 64 64)) (define b64 (randn 64 64))
  (lambda () (+ a8 b8) (@ a64 b64) (relu a64) (sum b64)))

(define (inplace-mix)
  (define c8 (zeros 8 8)) (define b8 (randn 8 8))
  (define a64 (randn 64 64)) (define b64 (randn 64 64))
  (define c64 (zeros 64 64)) (define one64 (ones 64 64))
  (lambda ()
    (add-tensor! c8 b8 1.0)
    (copy! c64 a64 #f)
    (addcmul! c64 a64 b64 1.0)
    (mul-tensor! c64 one64)))

(define (spin n)
  (let loop ([i 0] [acc 0])
    (if (fx= i n) acc (loop (fx+ i 1) (fxand (fx+ acc i) 1023)))))

;; calibrated on the lab host: the real mix is ~62 us and ~30 KB per
;; iteration (counting phantom bytes); spin 20000 is ~41 us.
(define cons-words (make-parameter 3750))
(define spin-iters (make-parameter 30000))

(define (cons-mix)
  (define n (cons-words))
  (define k (spin-iters))
  (define sink (box #f))
  (lambda ()
    (for ([_ (in-range 4)])
      (set-box! sink (make-vector (quotient n 4) 0)))
    (set-box! sink (spin k))))

(define (spin-mix)
  (define k (spin-iters))
  (define sink (box 0))
  (lambda () (set-box! sink (spin k))))

(require (only-in racket/fixnum fx= fx+ fxand))

(define mixes (hash "real" real-mix "inplace" inplace-mix "cons" cons-mix "spin" spin-mix))

;; --- main ---
(define (training-step)
  (define net (Sequential (Linear 784 256) relu (Linear 256 10)))
  (define opt (sgd (parameters net) #:lr 0.01))
  (define xs (randn 64 784))
  (define ys (tensor (for/list ([i (in-range 64)]) (modulo i 10)) #:dtype 'int64))
  (lambda ()
    (zero-grads! opt)
    (backward! (cross-entropy (net xs) ys))
    (step! opt)))

(define (now) (current-inexact-monotonic-milliseconds))

(define (run n mix-name state seconds)
  (define make-mix (hash-ref mixes mix-name))
  (define go? (box #f))
  (define stop? (box #f))
  (define counts (make-vector n 0))
  (define failure (box #f))
  (define tids (make-vector n #f))
  (define step! (and (eq? state 'train) (training-step)))
  (when step! (for ([_ (in-range 20)]) (step!)))
  (define custodian (make-custodian-at-root))
  (define ms (for/list ([_ (in-range n)]) (make-mix)))
  (define workers
    (parameterize ([current-custodian custodian])
      (for/list ([mix (in-list ms)] [i (in-naturals)])
        (thread (lambda ()
                  (vector-set! tids i (gettid))
                  (with-handlers ([(lambda (_) #t) (lambda (e) (set-box! failure e))])
                  (let loop ([k 0])
                    (cond
                      [(not (unbox go?)) (loop k)]
                      [(unbox stop?) (void)]
                      [else (mix) (vector-set! counts i (add1 k)) (loop (add1 k))]))))
                #:pool 'own))))
  (let wait () (unless (for/and ([t (in-vector tids)]) t) (sleep 0.01) (wait)))
  (collect-garbage)
  (define recv (make-log-receiver (current-logger) 'debug 'GC))
  (define steps-ms (make-flvector 200000 0.0))
  (define main-tid (gettid))
  (define load0 (loadavg))
  (define fin0 (cdr (assq 'runs (finalizer-diagnostics))))
  (define pd0 (finalizer-diagnostics))
  (diag-reset!)
  (define cas0 (cas-snapshot))
  (define gc0 (gc-acc-snapshot))
  (define gcms0 (current-gc-milliseconds))
  (define pv0 (make-vector 12 0)) (vector-set-performance-stats! pv0)
  (define s0 (sched-all))
  (define t0 (now))
  (set-box! go? #t)
  (define deadline (+ t0 (* 1000.0 seconds)))
  (define steps
    (cond
      [step!
       (let loop ([k 0])
         (define a (now))
         (cond
           [(>= a deadline) k]
           [else (step!)
                 (when (< k 200000) (flvector-set! steps-ms k (- (now) a)))
                 (loop (add1 k))]))]
      [else (sync (alarm-evt deadline #t)) 0]))
  (define t1 (now))
  (define s1 (sched-all))
  (define pv1 (make-vector 12 0)) (vector-set-performance-stats! pv1)
  (define gcms1 (current-gc-milliseconds))
  (define gc1 (gc-acc-snapshot))
  (define cas1 (cas-snapshot))
  (define diag (diag-snapshot))
  (define fin1 (cdr (assq 'runs (finalizer-diagnostics))))
  (define pd1 (finalizer-diagnostics))
  (define ops (for/sum ([c (in-vector counts)]) c))
  (set-box! stop? #t)
  (for ([w (in-list workers)]) (unless (sync/timeout 30 w) (kill-thread w)))
  (define load1 (loadavg))
  (when (unbox failure) (raise (unbox failure)))
  (define gcs
    (let loop ([acc '()])
      (match (sync/timeout 0 recv)
        [#f (reverse acc)]
        [(vector _ _ data _) (loop (if (and (vector? (struct->vector data))) (cons data acc) acc))])))
  (define gc-events
    (for/list ([d (in-list gcs)])
      (define v (struct->vector d))
      ;; #(struct:gc-info mode pre pre-admin code post post-admin sp ep st et)
      (list (vector-ref v 1) (- (vector-ref v 10) (vector-ref v 9))
            (- (vector-ref v 8) (vector-ref v 7)) (vector-ref v 2) (vector-ref v 5))))
  (define wall (- t1 t0))
  (define step-list (for/list ([i (in-range (min steps 200000))]) (flvector-ref steps-ms i)))
  (define sorted (sort step-list <))
  (define (pct p) (if (null? sorted) 0 (list-ref sorted (min (sub1 (length sorted)) (inexact->exact (floor (* p (length sorted))))))))
  (define (sched-delta tid)
    (define a (hash-ref s0 tid #f)) (define b (hash-ref s1 tid #f))
    (and a b (map - b a)))
  (define worker-tids (vector->list tids))
  (define other-tids (filter (lambda (t) (not (or (= t main-tid) (memv t worker-tids))))
                             (hash-keys s1)))
  (define (sum-sched ts)
    (for/fold ([acc '(0 0 0)]) ([t (in-list ts)])
      (define d (sched-delta t))
      (if d (map + acc d) acc)))
  (hasheq 'trip (vm-eval '(collect-trip-bytes))
          'finalizer_pool (or (getenv "RKTORCH_DIAG439_FINALIZERS") "ffi")
          'no_phantom (and (getenv "RKTORCH_DIAG439_NOPHANTOM") #t)
          'n n 'mix mix-name 'state (symbol->string state) 'seconds seconds
          'diag diag-enabled?
          'load_before load0 'load_after load1
          'wall_ms wall
          'main_steps steps 'main_steps_per_s (/ steps (/ wall 1000.0))
          'ops_per_s (/ (* 4 ops) (/ wall 1000.0))
          'step_ms_sum (apply + step-list)
          'step_ms_p50 (pct 0.5) 'step_ms_p90 (pct 0.9) 'step_ms_p99 (pct 0.99)
          'step_ms_max (if (null? sorted) 0 (last sorted))
          'gc_handler (map - gc1 gc0)
          'gc_events_n (length gc-events)
          'gc_events_wall_ms (for/sum ([e (in-list gc-events)]) (cadr e))
          'gc_events_cpu_ms (for/sum ([e (in-list gc-events)]) (caddr e))
          'gc_modes (for/fold ([h (hasheq)]) ([e (in-list gc-events)])
                      (hash-update h (car e) add1 0))
          'gc_major_wall_ms (for/sum ([e (in-list gc-events)] #:when (eq? (car e) 'major)) (cadr e))
          'gc_count_delta (- (vector-ref pv1 3) (vector-ref pv0 3))
          'gc_ms_delta (- gcms1 gcms0)
          'finalizer_runs (- fin1 fin0)
          'pressure_before (for/hasheq ([p (in-list pd0)] #:when (number? (cdr p))) (values (car p) (cdr p)))
          'pressure_after (for/hasheq ([p (in-list pd1)] #:when (number? (cdr p))) (values (car p) (cdr p)))
          'sites (for/hasheq ([(k v) (in-hash diag)]) (values k v))
          'cas (for/hasheq ([(k v) (in-hash cas1)])
                 (values k (map - v (hash-ref cas0 k))))
          'sched_main (sched-delta main-tid)
          'sched_workers (for/list ([t (in-list worker-tids)]) (sched-delta t))
          'sched_others (sum-sched other-tids)
          'n_tasks (length (hash-keys s1))))

(module+ main
  (define args (current-command-line-arguments))
  (match-define (vector n-s mix state-s seconds-s rest ...) args)
  (for ([kv (in-list rest)])
    (match (string-split kv "=")
      [(list "cons-words" v) (cons-words (string->number v))]
      [(list "spin-iters" v) (spin-iters (string->number v))]
      [(list "trip" v) (vm-eval `(collect-trip-bytes ,(string->number v)))]
      [_ (void)]))
  (define r (run (string->number n-s) mix (string->symbol state-s) (string->number seconds-s)))
  (write-json r)
  (newline))

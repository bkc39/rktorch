#lang racket/base

(require (only-in profile/analyzer analyze-samples node-id node-src node-total profile-nodes
                  profile-total-time)
         (only-in profile/sampler create-sampler)
         (only-in racket/file file->string)
         (only-in racket/list filter-map)
         (only-in racket/match match match-define)
         (only-in racket/os getpid)
         (only-in racket/string string-contains? string-split)
         (only-in threading ~>)
         (only-in torch/foreign/raw/collector finalizer-runs)
         (only-in "harness.rkt" clock-ms load-average rotate)
         (only-in "op-cost-layers.rkt" ladder pure-overhead-ops side-layers)
         (only-in "op-cost-ops.rkt"
                  build-case crossing-thunk crossings-per-call op-case-layer op-case-layers
                  op-case-name tensors-per-call)
         (only-in "ops.rkt" settle-native-memory!)
         (only-in "threads.rkt" scaling-run))

(provide conditions
         (struct-out condition)
         sample-condition
         crossing-record
         sampler-record
         sampler-pieces
         piece-share
         main-cpu-ms
         run-rep
         warm-up!)

;; `make` answers the thunk a thread runs: main's is the case built once, and
;; each worker builds its own, so no two threads share a case's state.
(struct condition (op layer threads make) #:transparent)

(define (layers-of c)
  (filter (lambda (l) (op-case-layer c l)) (append ladder side-layers)))

(define thread-layers '(raw registered accounted checked facade))

(define (conditions cases threads)
  (append
   (for*/list ([c (in-list cases)] [l (in-list (layers-of c))])
     (define thunk (op-case-layer c l))
     (condition (op-case-name c) l 0 (lambda () thunk)))
   (for*/list ([n (in-list threads)]
               [c (in-list cases)]
               [l (in-list (layers-of c))]
               #:when (or (memq (op-case-name c) pure-overhead-ops) (eq? l 'facade))
               #:when (memq l thread-layers))
     (define name (op-case-name c))
     (condition name l n (lambda () (op-case-layer (build-case name) l))))))

(define (main-loop thunk seconds)
  (define deadline (+ (clock-ms) (* 1000.0 seconds)))
  (define t0 (clock-ms))
  (let loop ([n 0])
    (cond
      [(>= (clock-ms) deadline) (values n (- (clock-ms) t0))]
      [else (thunk) (loop (add1 n))])))

(define (pool-loop n make seconds)
  (define run (scaling-run n 'idle #:seconds seconds #:mix make #:ops-per-mix 1))
  (values (hash-ref run 'mixes) (* 1000.0 (hash-ref run 'elapsed_s))))

;; Green threads, the finalizer thread among them, run on the process's
;; first OS thread, whose task id is the pid; Linux's USER_HZ is 100.
(define main-task-stat (format "/proc/self/task/~a/stat" (getpid)))

(define (main-cpu-ms)
  (and (file-exists? main-task-stat)
       (match (~> main-task-stat file->string (regexp-match #rx"\\) (.*)$" _))
         [(list _ fields)
          (define ticks (map string->number (string-split fields)))
          (* 10.0 (+ (list-ref ticks 11) (list-ref ticks 12)))]
         [_ #f])))

(define (counters)
  (hasheq 'gc (current-gc-milliseconds)
          'heap (current-memory-use 'cumulative)
          'finalizers (finalizer-runs)
          'main-cpu (main-cpu-ms)))

(define (sample-condition c seconds #:rep rep)
  (match-define (condition op layer n make) c)
  (settle-native-memory!)
  (define load (load-average))
  (define before (counters))
  (define-values (ops elapsed-ms)
    (if (zero? n) (main-loop (make) seconds) (pool-loop n make seconds)))
  (define after (counters))
  (define (moved key)
    (define b (hash-ref before key))
    (define a (hash-ref after key))
    (and a b (- a b)))
  (define main-ms (and (positive? n) (moved 'main-cpu)))
  (define (per-op x) (and x (positive? ops) (exact->inexact (/ x ops))))
  (define thread-ms (* elapsed-ms (max 1 n)))
  (hasheq 'kind "op"
          'impl "rktorch"
          'op (symbol->string op)
          'layer (symbol->string layer)
          'threads n
          'rep rep
          'ops ops
          'elapsed_ms elapsed-ms
          'us_per_op (~> (or main-ms 0) (+ thread-ms) (* 1000.0) per-op)
          'thread_us_per_op (per-op (* 1000.0 thread-ms))
          'main_cpu_ms main-ms
          'ops_per_s (and (positive? elapsed-ms) (/ (* 1000.0 ops) elapsed-ms))
          'heap_bytes_per_op (per-op (moved 'heap))
          'finalizer_runs (moved 'finalizers)
          'finalized_per_op (per-op (moved 'finalizers))
          'gc_ms (moved 'gc)
          'load load))

(define (crossing-record rep #:calls [calls 200000])
  (define f (crossing-thunk))
  (define t0 (clock-ms))
  (for ([_ (in-range calls)]) (f))
  (hasheq 'kind "crossing" 'impl "rktorch" 'rep rep
          'us_per_crossing (~> (clock-ms) (- t0) (* 1000.0) (/ calls))
          'load (load-average)))

(define (contract-node? n)
  (define src (node-src n))
  (and src (string-contains? (format "~a" src) "racket/contract")))

(define ((named . names) n)
  (and (memq (node-id n) names) #t))

;; the drain the pressure check's yield runs is finalization's, not the check's
(define sampler-pieces
  (hasheq 'accounting (named 'account! 'record-allocation!)
          'pressure (named 'collect-under-pressure!)
          'finalizer (named 'register-tensor! 'yield-to-finalizers!)
          'shape (named 'handle-shape 'wrap-tensor)
          'contract contract-node?))

(define (piece-share totals total)
  (cond
    [(or (null? totals) (not (positive? total))) 0.0]
    [else (~> (apply max totals) (/ total) exact->inexact)]))

(define (sampler-record c layer seconds)
  (define thunk (op-case-layer c layer))
  (settle-native-memory!)
  (define sampler (create-sampler (current-thread) 0.001))
  (define-values (ops elapsed-ms) (main-loop thunk seconds))
  (sampler 'stop)
  (define profile (analyze-samples (sampler 'get-snapshots)))
  (define nodes (profile-nodes profile))
  (define total (profile-total-time profile))
  (hasheq 'kind "sampler"
          'impl "rktorch"
          'op (symbol->string (op-case-name c))
          'layer (symbol->string layer)
          'ops ops
          'us_per_op (and (positive? ops) (/ (* 1000.0 elapsed-ms) ops))
          'total_ms (exact->inexact total)
          'shares (for/hasheq ([(piece matches?) (in-hash sampler-pieces)])
                    (values piece
                            (piece-share (filter-map (lambda (n) (and (matches? n) (node-total n)))
                                                     nodes)
                                         total)))
          'load (load-average)))

(define (warm-up! conds)
  (for ([c (in-list conds)] #:when (zero? (condition-threads c)))
    (main-loop ((condition-make c)) 0.1)))

(define (counts-record cases)
  (for*/list ([c (in-list cases)] [(l thunk) (in-hash (op-case-layers c))])
    (hasheq 'kind "counts" 'impl "rktorch"
            'op (symbol->string (op-case-name c)) 'layer (symbol->string l)
            'crossings_per_op (exact->inexact (crossings-per-call thunk))
            'tensors_per_op (exact->inexact (tensors-per-call thunk)))))

(define (run-rep cases conds rep #:seconds seconds #:thread-seconds thread-seconds #:emit emit)
  (when (zero? rep)
    (for-each emit (counts-record cases)))
  (emit (crossing-record rep))
  (for ([c (in-list (rotate conds rep))])
    (emit (sample-condition c (if (zero? (condition-threads c)) seconds thread-seconds)
                            #:rep rep))))

(module+ test
  (require (only-in json jsexpr?)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "op-cost-ops.rkt" ladder-ops))

  (define (sleeper ms)
    (condition 'sleep 'raw 0 (lambda () (lambda () (sleep (/ ms 1000.0))))))

  (test-case "threaded conditions take every layer of the adds, and the facade of the rest"
    (define conds (conditions (ladder-ops '(scalar-add narrow)) '(2)))
    (define threaded
      (for/list ([c (in-list conds)] #:when (= 2 (condition-threads c)))
        (list (condition-op c) (condition-layer c))))
    (check-equal? threaded
                  '((scalar-add raw) (scalar-add registered) (scalar-add accounted)
                    (scalar-add checked) (scalar-add facade) (narrow facade)))
    (check-equal? (length (filter (lambda (c) (zero? (condition-threads c))) conds)) 23))

  (test-case "a threaded condition builds a fresh case per worker; main's is built once"
    (define conds (conditions (ladder-ops '(scalar-add)) '(2)))
    (define (made threads)
      (define c (findf (lambda (x) (and (= threads (condition-threads x))
                                        (eq? 'facade (condition-layer x))))
                       conds))
      (eq? ((condition-make c)) ((condition-make c))))
    (check-true (made 0))
    (check-false (made 2)))

  (test-case "a sample prices a known op: 2 ms a call, on main and per thread"
    (call-with-process-deadline
     120
     (lambda ()
       (for ([n (in-list '(0 2))])
         (define r (sample-condition (struct-copy condition (sleeper 2) [threads n]) 0.2 #:rep 3))
         (check-equal? (hash-ref r 'threads) n)
         (check-equal? (hash-ref r 'rep) 3)
         (check-true (< 1900 (hash-ref r 'us_per_op) 6000) (format "~a" r))
         (check-true (real? (hash-ref r 'heap_bytes_per_op)))
         (check-true (jsexpr? r) (format "~a" r))))))

  (test-case "main's OS thread CPU time grows while main spins, not while it sleeps"
    (when (main-cpu-ms)
      (define t0 (main-cpu-ms))
      (sleep 0.3)
      (define t1 (main-cpu-ms))
      (define deadline (+ (clock-ms) 300))
      (let spin () (when (< (clock-ms) deadline) (spin)))
      (define t2 (main-cpu-ms))
      (check-true (< (- t1 t0) 100) (format "asleep ~a ms" (- t1 t0)))
      (check-true (> (- t2 t1) 150) (format "spinning ~a ms" (- t2 t1)))))

  ;; Without the yield, registered and retry drain 45% and 55% in window;
  ;; with it, 90% and 97%.
  (test-case "every layer that registers drains its finalizers inside its own window"
    (call-with-process-deadline
     300
     (lambda ()
       (define c (car (ladder-ops '(scalar-add))))
       (for ([l (in-list '(registered retry accounted checked wrapped unchecked facade))])
         (define layer-condition (condition 'scalar-add l 0 (lambda () (op-case-layer c l))))
         (sample-condition layer-condition 0.1 #:rep 0)
         (define r (sample-condition layer-condition 0.5 #:rep 0))
         (define drained (/ (hash-ref r 'finalizer_runs) (hash-ref r 'ops)))
         (check-true (> drained 3/4) (format "~a drained ~a" l (exact->inexact drained)))))))

  (test-case "a threaded sample records main's finalizer runs and CPU time beside the workers'"
    (call-with-process-deadline
     120
     (lambda ()
       (define c (car (ladder-ops '(scalar-add))))
       (define r (sample-condition (findf (lambda (x) (and (= 2 (condition-threads x))
                                                           (eq? 'registered (condition-layer x))))
                                          (conditions (list c) '(2)))
                                   0.3 #:rep 0))
       (check-true (positive? (hash-ref r 'finalizer_runs)))
       (check-true (jsexpr? r) (format "~a" r))
       (when (hash-ref r 'main_cpu_ms)
         (check-= (hash-ref r 'us_per_op)
                  (+ (hash-ref r 'thread_us_per_op)
                     (/ (* 1000.0 (hash-ref r 'main_cpu_ms)) (hash-ref r 'ops)))
                  1e-6)))))

  (test-case "a sampler record of the contracted add is one JSON row"
    (define r (sampler-record (car (ladder-ops '(scalar-add))) 'contracted 0.3))
    (check-equal? (hash-ref r 'layer) "contracted")
    (check-true (jsexpr? r) (format "~a" r)))

  (test-case "a sampler piece's share is its outermost function's time over the total"
    (check-equal? (piece-share '(30 10 20) 100) 0.3)
    (check-equal? (piece-share '() 100) 0.0)
    (check-equal? (piece-share '(5) 0) 0.0)))

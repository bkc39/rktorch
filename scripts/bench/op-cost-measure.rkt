#lang racket/base

(require (only-in profile/analyzer analyze-samples node-id node-src node-total profile-nodes
                  profile-total-time)
         (only-in profile/sampler create-sampler)
         (only-in racket/list filter-map)
         (only-in racket/string string-contains?)
         (only-in "harness.rkt" clock-ms load-average rotate)
         (only-in "op-cost-layers.rkt" ladder side-layers)
         (only-in "op-cost-ops.rkt"
                  crossing-thunk crossings-per-call op-case-layer op-case-layers op-case-name
                  tensors-per-call)
         (only-in "ops.rkt" settle-native-memory!)
         (only-in "threads.rkt" scaling-run))

(provide conditions
         (struct-out condition)
         sample-condition
         crossing-record
         sampler-record
         sampler-pieces
         piece-share
         run-rep
         warm-up!)

(struct condition (op layer threads thunk) #:transparent)

(define (layers-of c)
  (filter (lambda (l) (op-case-layer c l)) (append ladder side-layers)))

(define thread-layers '(raw registered accounted checked facade))

(define (conditions cases threads)
  (append
   (for*/list ([c (in-list cases)] [l (in-list (layers-of c))])
     (condition (op-case-name c) l 0 (op-case-layer c l)))
   (for*/list ([n (in-list threads)]
               [c (in-list cases)]
               [l (in-list (layers-of c))]
               #:when (or (memq (op-case-name c) '(scalar-add add-8x8))
                          (eq? l 'facade))
               #:when (memq l thread-layers))
     (condition (op-case-name c) l n (op-case-layer c l)))))

(define (main-loop thunk seconds)
  (define deadline (+ (clock-ms) (* 1000.0 seconds)))
  (define t0 (clock-ms))
  (let loop ([n 0])
    (cond
      [(>= (clock-ms) deadline) (values n (- (clock-ms) t0))]
      [else (thunk) (loop (add1 n))])))

(define (pool-loop n thunk seconds)
  (define run (scaling-run n 'idle #:seconds seconds #:mix (lambda () thunk) #:ops-per-mix 1))
  (values (hash-ref run 'mixes) (* 1000.0 (hash-ref run 'elapsed_s))))

(define (sample-condition cond-v seconds #:rep rep)
  (settle-native-memory!)
  (define thunk (condition-thunk cond-v))
  (define n (condition-threads cond-v))
  (define load (load-average))
  (define gc0 (current-gc-milliseconds))
  (define heap0 (current-memory-use 'cumulative))
  (define-values (ops elapsed-ms)
    (if (zero? n) (main-loop thunk seconds) (pool-loop n thunk seconds)))
  (define heap1 (current-memory-use 'cumulative))
  (define gc1 (current-gc-milliseconds))
  (define per-op (and (positive? ops) (/ 1.0 ops)))
  (hasheq 'kind "op"
          'impl "rktorch"
          'op (symbol->string (condition-op cond-v))
          'layer (symbol->string (condition-layer cond-v))
          'threads n
          'rep rep
          'ops ops
          'elapsed_ms elapsed-ms
          'us_per_op (and per-op (* 1000.0 elapsed-ms (max 1 n) per-op))
          'ops_per_s (and (positive? elapsed-ms) (/ (* 1000.0 ops) elapsed-ms))
          'heap_bytes_per_op (and per-op (* per-op (- heap1 heap0)))
          'gc_ms (- gc1 gc0)
          'load load))

(define (crossing-record rep #:calls [calls 200000])
  (define f (crossing-thunk))
  (define t0 (clock-ms))
  (for ([_ (in-range calls)]) (f))
  (hasheq 'kind "crossing" 'impl "rktorch" 'rep rep
          'us_per_crossing (/ (* 1000.0 (- (clock-ms) t0)) calls)
          'load (load-average)))

(define (contract-node? n)
  (define src (node-src n))
  (and src
       (string-contains? (format "~a" src) "racket/contract")))

(define ((named . names) n)
  (and (memq (node-id n) names) #t))

(define sampler-pieces
  (list (cons 'accounting (named 'account! 'record-allocation!))
        (cons 'pressure (named 'collect-under-pressure!))
        (cons 'finalizer (named 'register-tensor!))
        (cons 'shape (named 'handle-shape 'wrap-tensor))
        (cons 'contract contract-node?)))

(define (piece-share totals total)
  (cond
    [(or (null? totals) (not (positive? total))) 0.0]
    [else (exact->inexact (/ (apply max totals) total))]))

(define (sampler-record c layer seconds)
  (define thunk (op-case-layer c layer))
  (settle-native-memory!)
  (define sampler (create-sampler (current-thread) 0.001))
  (define-values (ops _elapsed) (main-loop thunk seconds))
  (sampler 'stop)
  (define p (analyze-samples (sampler 'get-snapshots)))
  (define nodes (profile-nodes p))
  (define total (profile-total-time p))
  (hasheq 'kind "sampler"
          'impl "rktorch"
          'op (symbol->string (op-case-name c))
          'layer (symbol->string layer)
          'ops ops
          'total_ms total
          'shares (for/hasheq ([piece (in-list sampler-pieces)])
                    (values (car piece)
                            (piece-share (filter-map (lambda (n) (and ((cdr piece) n)
                                                                      (node-total n)))
                                                     nodes)
                                         total)))
          'load (load-average)))

(define (warm-up! conds)
  (for ([c (in-list conds)] #:when (zero? (condition-threads c)))
    (main-loop (condition-thunk c) 0.1)))

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
  (require rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "op-cost-ops.rkt" ladder-ops))

  (test-case "threaded conditions take every layer of the adds, and the facade of the rest"
    (define conds (conditions (ladder-ops '(scalar-add narrow)) '(2)))
    (define threaded
      (for/list ([c (in-list conds)] #:when (= 2 (condition-threads c)))
        (list (condition-op c) (condition-layer c))))
    (check-equal? threaded
                  '((scalar-add raw) (scalar-add registered) (scalar-add accounted)
                    (scalar-add checked) (scalar-add facade) (narrow facade)))
    (check-equal? (length (filter (lambda (c) (zero? (condition-threads c))) conds)) 17))

  (test-case "a sample counts the ops the threads ran, and prices them per thread"
    (call-with-process-deadline
     120
     (lambda ()
       (for ([n (in-list '(0 2))])
         (define r (sample-condition (condition 'noop 'raw n void) 0.05 #:rep 3))
         (check-equal? (hash-ref r 'threads) n)
         (check-equal? (hash-ref r 'rep) 3)
         (check-true (positive? (hash-ref r 'ops)))
         (check-= (hash-ref r 'us_per_op)
                  (/ (* 1000.0 (hash-ref r 'elapsed_ms) (max 1 n)) (hash-ref r 'ops))
                  1e-6)
         (check-true (real? (hash-ref r 'heap_bytes_per_op)))))))

  (test-case "a sampler piece's share is its outermost function's time over the total"
    (check-equal? (piece-share '(30 10 20) 100) 0.3)
    (check-equal? (piece-share '() 100) 0.0)
    (check-equal? (piece-share '(5) 0) 0.0)))

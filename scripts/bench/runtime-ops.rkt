#lang racket/base

(require (only-in ffi/unsafe _fun)
         (only-in racket/list append*)
         (only-in racket/match match match-define)
         (only-in torch randn)
         (only-in torch/foreign/raw/elementwise tr-add/raw)
         (only-in torch/foreign/raw/linalg tr-matmul/raw)
         (only-in torch/foreign/raw/memory tensor-allocator)
         (only-in torch/foreign/raw/syntax _Tensor _Tensor/null define-torch)
         (only-in "harness.rkt" clock-ms)
         (only-in "runtime.rkt"
                  allocate-until-deadline cell cell-ffi cell-main cell-workload duration-run)
         (only-in "runtime-work.rkt" poll!)
         (only-in "threads.rkt" training-step))

(provide op-names
         ops-cells
         run-op-cell)

(define-torch tr-add/collect-safe
  (_fun #:blocking? #t (a : _Tensor) (b : _Tensor) -> _Tensor/null)
  #:c-id tr_add
  #:wrap tensor-allocator)

(define-torch tr-matmul/collect-safe
  (_fun #:blocking? #t (a : _Tensor) (b : _Tensor) -> _Tensor/null)
  #:c-id tr_matmul
  #:wrap tensor-allocator)

(struct op (name size plain collect-safe))

(define ops
  (list (op 'add32 32 tr-add/raw tr-add/collect-safe)
        (op 'mm256 256 tr-matmul/raw tr-matmul/collect-safe)
        (op 'mm1024 1024 tr-matmul/raw tr-matmul/collect-safe)))

(define op-names (map op-name ops))

(define (op-named name)
  (findf (lambda (o) (eq? (op-name o) name)) ops))

(define (make-op name ffi)
  (match-define (op _ size plain collect-safe) (op-named name))
  (define a (randn size size))
  (define b (randn size size))
  (match ffi
    ['plain (lambda () (plain a b))]
    ['poll (lambda () (begin0 (plain a b) (poll!)))]
    ['collect-safe (lambda () (collect-safe a b))]))

(define (training-until)
  (define step! (training-step))
  (lambda (deadline)
    (let loop ([n 0] [gaps '()])
      (define t0 (clock-ms))
      (cond
        [(>= t0 deadline) (values n gaps)]
        [else
         (step!)
         (loop (add1 n) (cons (- (clock-ms) t0) gaps))]))))

(define (run-op-cell c)
  (define main-loop
    (match (cell-main c)
      ['alloc allocate-until-deadline]
      ['train (training-until)]))
  (duration-run c (lambda () (make-op (cell-workload c) (cell-ffi c))) main-loop))

(define (ops-cells #:workers ks
                   #:trips trips
                   #:seconds seconds
                   #:ops [names op-names]
                   #:ffis [ffis '(plain poll collect-safe)]
                   #:mains [mains '(alloc train)])
  (append*
   (for*/list ([trip (in-list trips)] [main (in-list mains)])
     (cons (cell #:group 'ops #:workload 'none #:main main #:k 0 #:trip trip #:seconds seconds)
           (for*/list ([name (in-list names)] [ffi (in-list ffis)] [k (in-list ks)])
             (cell #:group 'ops #:workload name #:main main #:k k #:trip trip #:ffi ffi
                   #:seconds seconds))))))

(module+ test
  (require (only-in racket/port open-output-nowhere)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "runtime.rkt" cell-records))

  (test-case "every op arm runs in workers beside main allocating or training"
    (call-with-process-deadline
     180
     (lambda ()
       (define records
         (cell-records (ops-cells #:workers '(1) #:trips '(#f) #:seconds 0.2
                                  #:ops '(add32 mm256))
                       run-op-cell
                       #:rounds 1 #:warmup 0 #:meta (hasheq) #:max-load #f
                       #:log (open-output-nowhere)))
       (check-equal? (for/list ([r (in-list records)])
                       (list (hash-ref r 'main) (hash-ref r 'workload) (hash-ref r 'ffi)
                             (hash-ref r 'k)))
                     (for*/list ([main (in-list '("alloc" "train"))]
                                 [arm (in-list (cons '("none" "none" 0)
                                                     (for*/list ([op (in-list '("add32" "mm256"))]
                                                                 [ffi (in-list '("plain" "poll"
                                                                                 "collect-safe"))])
                                                       (list op ffi 1))))])
                       (cons main arm)))
       (for ([r (in-list records)])
         (define (only key) (car (hash-ref r key)))
         (define where (hash-ref r 'variant))
         (check-equal? (hash-ref r 'group) "ops")
         (check-equal? (hash-ref r 'unit)
                       (if (equal? (hash-ref r 'main) "train") "steps/s" "M allocs/s"))
         (check-true (positive? (only 'main_rate)) where)
         (check-pred real? (only 'pause_p99_ms) where)
         (check-equal? (only 'stopped) (hash-ref r 'k) where)
         (check-equal? (positive? (only 'worker_ops_per_s)) (positive? (hash-ref r 'k)) where)
         (check-equal? (real? (only 'us_per_op)) (positive? (hash-ref r 'k)) where))))))

#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe _fun)
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
         (only-in "threads.rkt" training-step)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide op-names
         ops-cells
         run-op-cell)

(struct calls (plain collect-safe))

(define-syntax-parse-rule (binary-calls plain:id c-id:id)
  (let ()
    (define-torch collect-safe
      (_fun #:blocking? #t (a : _Tensor) (b : _Tensor) -> _Tensor/null)
      #:c-id c-id
      #:wrap tensor-allocator)
    (calls plain collect-safe)))

(define add (binary-calls tr-add/raw tr_add))
(define matmul (binary-calls tr-matmul/raw tr_matmul))

(struct op (name size calls))

(define ops
  (list (op 'add32 32 add)
        (op 'mm256 256 matmul)
        (op 'mm1024 1024 matmul)))

(define op-names (map op-name ops))

(define (op-named name)
  (findf (lambda (o) (eq? (op-name o) name)) ops))

(define (make-op name ffi)
  (match-define (op _ size (calls plain collect-safe)) (op-named name))
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
                     '(("alloc" "none" "none" 0)
                       ("alloc" "add32" "plain" 1)
                       ("alloc" "add32" "poll" 1)
                       ("alloc" "add32" "collect-safe" 1)
                       ("alloc" "mm256" "plain" 1)
                       ("alloc" "mm256" "poll" 1)
                       ("alloc" "mm256" "collect-safe" 1)
                       ("train" "none" "none" 0)
                       ("train" "add32" "plain" 1)
                       ("train" "add32" "poll" 1)
                       ("train" "add32" "collect-safe" 1)
                       ("train" "mm256" "plain" 1)
                       ("train" "mm256" "poll" 1)
                       ("train" "mm256" "collect-safe" 1)))
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
         (check-equal? (real? (only 'us_per_op)) (positive? (hash-ref r 'k)) where)))))

  (test-case "main's p99 pause beside a 1024² matmul worker falls tenfold in collect-safe calls"
    (call-with-process-deadline
     120
     (lambda ()
       (define (p99 ffi)
         (define c (cell #:group 'ops #:workload 'mm1024 #:main 'alloc #:k 1 #:ffi ffi
                         #:seconds 1.0))
         (hash-ref (run-op-cell c) 'pause_p99_ms))
       (define plain (p99 'plain))
       (define safe (p99 'collect-safe))
       (check-true (< (* 10 safe) plain) (format "collect-safe ~a ms, plain ~a ms" safe plain))))))

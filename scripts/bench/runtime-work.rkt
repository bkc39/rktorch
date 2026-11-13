#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe _fun _int _uint32 get-ffi-obj)
         (only-in ffi/unsafe/os-thread make-os-semaphore os-semaphore-post os-semaphore-wait)
         (only-in ffi/unsafe/vm vm-eval vm-primitive)
         (only-in racket/match match match-define)
         (only-in racket/math exact-round)
         (only-in racket/place place-channel-get place-channel-put)
         (only-in racket/unsafe/ops unsafe-fx* unsafe-fx+ unsafe-fxand)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide workload-names
         work-units
         run-work
         allocate-until
         with-trip-bytes
         native-calls
         native-call
         poll!
         place-worker)

(define (cpu n)
  (let loop ([i 0] [acc 0])
    (if (= i n) acc (loop (add1 i) (unsafe-fxand (unsafe-fx+ acc (unsafe-fx* i 7)) #xFFFFF)))))

(define (alloc n)
  (let loop ([i 0] [keep #f])
    (if (= i n) keep (loop (add1 i) (make-vector 8 i)))))

(define ring-size 20000)

(define (alloc-retain n)
  (define ring (make-vector ring-size #f))
  (let loop ([i 0])
    (when (< i n)
      (vector-set! ring (modulo i ring-size) (make-vector 8 i))
      (loop (add1 i)))))

(define make-guardian (vm-eval 'make-guardian))

(define (drain! g)
  (when (g) (drain! g)))

(define (alloc-guarded n)
  (define g (make-guardian))
  (let loop ([i 0])
    (when (< i n)
      (g (make-vector 8 i))
      (when (zero? (unsafe-fxand i 1023)) (drain! g))
      (loop (add1 i))))
  (drain! g))

(define (sema n)
  (define s (make-semaphore 0))
  (for ([_ (in-range n)])
    (semaphore-post s)
    (semaphore-wait s)))

(define (os-sema n)
  (define s (make-os-semaphore))
  (for ([_ (in-range n)])
    (os-semaphore-post s)
    (os-semaphore-wait s)))

(define (cas n)
  (define b (box 0))
  (for ([_ (in-range n)])
    (let retry ()
      (define v (unbox b))
      (unless (box-cas! b v (add1 v)) (retry)))))

(struct workload (name run units))

(define workloads
  (list (workload 'cpu cpu 300000000)
        (workload 'alloc alloc 30000000)
        (workload 'alloc-retain alloc-retain 15000000)
        (workload 'alloc-guarded alloc-guarded 15000000)
        (workload 'sema sema 300000)
        (workload 'os-sema os-sema 3000000)
        (workload 'cas cas 30000000)))

(define workload-names (map workload-name workloads))

(define (workload-named name)
  (findf (lambda (w) (eq? (workload-name w) name)) workloads))

(define (work-units name scale)
  (max 1 (exact-round (* scale (workload-units (workload-named name))))))

(define (run-work name units)
  ((workload-run (workload-named name)) units))

(define sink (box #f))

(define (allocate-until stop?)
  (define t0 (current-inexact-monotonic-milliseconds))
  (let loop ([i 0] [keep #f] [last t0] [gaps '()])
    (cond
      [(positive? (unsafe-fxand i 4095)) (loop (add1 i) (make-vector 8 i) last gaps)]
      [else
       (define now (current-inexact-monotonic-milliseconds))
       (define seen (if (zero? i) gaps (cons (- now last) gaps)))
       (cond
         [(stop? now) (set-box! sink keep) (values i seen)]
         [else (loop (add1 i) (make-vector 8 i) now seen)])])))

(define (call-with-trip-bytes mb thunk)
  (match mb
    [#f (thunk)]
    [_
     (define before (vm-eval '(collect-trip-bytes)))
     (dynamic-wind (lambda () (vm-eval `(collect-trip-bytes ,(* mb 1024 1024))))
                   thunk
                   (lambda () (vm-eval `(collect-trip-bytes ,before))))]))

(define-syntax-parse-rule (with-trip-bytes mb:expr body:expr ...+)
  (call-with-trip-bytes mb (lambda () body ...)))

(define usleep/plain (get-ffi-obj "usleep" #f (_fun _uint32 -> _int)))
(define usleep/collect-safe (get-ffi-obj "usleep" #f (_fun #:blocking? #t _uint32 -> _int)))

(define native-calls '((usleep-2000 . 2000) (usleep-200 . 200)))

(define disable-interrupts! (vm-primitive 'disable-interrupts))
(define enable-interrupts! (vm-primitive 'enable-interrupts))

(define (poll!)
  (disable-interrupts!)
  (enable-interrupts!))

(define (native-call name ffi between)
  (define us (cdr (assq name native-calls)))
  (define call (if (eq? ffi 'collect-safe) usleep/collect-safe usleep/plain))
  (define poll? (eq? ffi 'poll))
  (define alloc? (eq? between 'alloc))
  (lambda ()
    (call us)
    (when poll? (poll!))
    (and alloc? (make-vector 8 0))))

(define (place-worker ch)
  (place-channel-put ch
                     (with-handlers ([exn:fail? (lambda (e) (list 'failed (exn-message e)))])
                       (match-define (list name units) (place-channel-get ch))
                       (place-channel-put ch 'ready)
                       (place-channel-get ch)
                       (run-work name units)
                       'done))
  (place-channel-get ch))

(module+ test
  (require rackunit)

  (test-case "every workload runs its units and returns"
    (for ([name (in-list workload-names)])
      (check-equal? (work-units name 1e-9) 1 (format "~a" name))
      (run-work name (work-units name 1e-5)))
    (check-equal? (work-units 'cpu 1) 300000000)
    (check-equal? (alloc 3) (make-vector 8 2)))

  (test-case "main allocates until told to stop, and keeps the gaps between clock reads"
    (define deadline (+ (current-inexact-monotonic-milliseconds) 50.0))
    (define-values (n gaps) (allocate-until (lambda (now) (>= now deadline))))
    (check-true (positive? n))
    (check-equal? (length gaps) (quotient n 4096))
    (check-true (andmap (lambda (g) (>= g 0.0)) gaps)))

  (test-case "a trip setting holds only inside its body"
    (define before (vm-eval '(collect-trip-bytes)))
    (with-trip-bytes 3
      (check-equal? (vm-eval '(collect-trip-bytes)) (* 3 1024 1024)))
    (check-equal? (vm-eval '(collect-trip-bytes)) before)
    (check-exn #rx"boom"
               (lambda () (with-trip-bytes 5 (error "boom"))))
    (check-equal? (vm-eval '(collect-trip-bytes)) before)
    (with-trip-bytes #f
      (check-equal? (vm-eval '(collect-trip-bytes)) before)))

  (test-case "a native call sleeps, polls or allocates as its arm says"
    (for* ([ffi (in-list '(plain poll collect-safe))] [between (in-list '(none alloc))])
      (define call (native-call 'usleep-200 ffi between))
      (check-equal? (call) (and (eq? between 'alloc) (make-vector 8 0)) (format "~a" ffi)))))

#lang racket/base

;; #262 question 3: register-finalizer, make-phantom-bytes and
;; set-phantom-bytes! called from parallel threads, plus the shared eq?-keyed
;; tables ffi/unsafe/alloc keeps its registrations in.

(require (only-in '#%foreign make-late-weak-hasheq)
         (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic call-as-uninterruptible)
         "common.rkt")

(define default-threads 8)

(define (vector-cas-add1! v i)
  (let retry ()
    (define old (vector-ref v i))
    (unless (vector-cas! v i old (add1 old))
      (retry))))

(define (collect-until done? #:limit-ms [limit-ms 60000])
  (define start (current-inexact-monotonic-milliseconds))
  (let loop ([rounds 1])
    (collect-garbage 'major)
    (sleep 0.02)
    (define elapsed (- (current-inexact-monotonic-milliseconds) start))
    (cond
      [(done?) (values rounds elapsed)]
      [(> elapsed limit-ms) (values rounds elapsed)]
      [else (loop (add1 rounds))])))

;; Every object's finalizer bumps its own slot, so a lost run reads 0 and a
;; doubled one reads 2. The first registration in the process is one of
;; these, made from 8 threads at once: that is the race finalizer.rkt
;; accepts by allowing several finalizer threads.
(define (finalizer-round per-thread #:protect protect #:threads [threads default-threads])
  (define total (* threads per-thread))
  (define runs (make-vector total 0))
  (define-values (ms _)
    (run-parallel threads
                  (lambda (t)
                    (for ([k (in-range per-thread)])
                      (define id (+ (* t per-thread) k))
                      (protect
                       (lambda ()
                         (register-finalizer (make-vector 2 id)
                                             (lambda (_v)
                                               (vector-cas-add1! runs id)))))))))
  (define (ran) (for/sum ([n (in-vector runs)]) n))
  (define-values (rounds drain-ms) (collect-until (lambda () (= (ran) total))))
  (define never (for/sum ([n (in-vector runs)]) (if (zero? n) 1 0)))
  (define twice (for/sum ([n (in-vector runs)]) (if (> n 1) 1 0)))
  (list total (fmt-ms ms) (fmt-rate (/ total (/ ms 1000.0)))
        (ran) never twice rounds (fmt-ms drain-ms)
        (if (and (zero? never) (zero? twice)) "yes" "NO")
        (load-average)))

(define (plain f) (f))
(define (break-disabled f) (parameterize-break #f (f)))

(define mib (* 1024 1024))

(define (settled-memory)
  (collect-garbage 'major)
  (collect-garbage 'major)
  (current-memory-use))

;; Phantom totals: the GC's count must rise by exactly what the threads
;; registered and fall back when they zero it, after any amount of churn.
(define (phantom-round count-per-thread size churn churn-size
                       #:threads [threads default-threads])
  (define base (settled-memory))
  (define-values (make-ms phantoms)
    (run-parallel threads
                  (lambda (_t)
                    (for/vector #:length count-per-thread
                                ([_ (in-range count-per-thread)])
                      (make-phantom-bytes size)))))
  (define held (- (settled-memory) base))
  (define expected (* threads count-per-thread size))
  (define-values (churn-ms _)
    (run-parallel threads
                  (lambda (t)
                    (define mine (list-ref phantoms t))
                    (for ([i (in-range churn)])
                        (define p (vector-ref mine (modulo i count-per-thread)))
                      (set-phantom-bytes! p churn-size)
                      (set-phantom-bytes! p size)))))
  (define after-churn (- (settled-memory) base))
  (define-values (zero-ms __)
    (run-parallel threads
                  (lambda (t)
                    (for ([p (in-vector (list-ref phantoms t))])
                      (set-phantom-bytes! p 0)))))
  (define after-zero (- (settled-memory) base))
  (list (format "~a x ~a x ~a KiB" threads count-per-thread (quotient size 1024))
        (format "~a x ~a MiB" (* threads churn) (quotient churn-size mib))
        (fmt-ms (/ expected mib 1.0))
        (fmt-ms (/ held mib 1.0))
        (fmt-ms (/ after-churn mib 1.0))
        (fmt-ms (/ after-zero mib 1.0))
        (fmt-rate (/ (* 2 threads churn) (/ churn-ms 1000.0)))
        (fmt-ms make-ms)
        (fmt-ms zero-ms)
        (load-average)))

;; Distinct keys per thread, so a correct table ends with every key present
;; once, then empty; a lost insertion or a corrupted bucket shows in the
;; counts.
(define (table-round label make-table per-thread #:protect protect
                     #:threads [threads default-threads])
  (define table (make-table))
  (define keys
    (for/vector ([_ (in-range (* threads per-thread))]) (make-vector 1 #f)))
  (define (each-key t f)
    (for ([k (in-range per-thread)])
      (define id (+ (* t per-thread) k))
      (protect (lambda () (f id (vector-ref keys id))))))
  (define-values (insert-ms _)
    (run-parallel threads
                  (lambda (t)
                    (each-key t (lambda (id key) (hash-set! table key id))))))
  (define count-after-insert (hash-count table))
  (define mismatched
    (for/sum ([(key id) (in-indexed keys)])
      (if (eqv? (hash-ref table key #f) id) 0 1)))
  (define-values (remove-ms __)
    (run-parallel threads
                  (lambda (t)
                    (each-key t (lambda (_id key) (hash-remove! table key))))))
  (list label
        (* threads per-thread)
        count-after-insert
        mismatched
        (hash-count table)
        (if (and (= count-after-insert (* threads per-thread))
                 (zero? mismatched)
                 (zero? (hash-count table)))
            "yes" "NO")
        (fmt-ms insert-ms)
        (fmt-ms remove-ms)
        (load-average)))

(define (in-indexed v)
  (in-parallel (in-vector v) (in-naturals)))

(module+ main
  (print-banner "Q3: finalizers, phantom bytes and eq? tables from 8 parallel threads")
  (print-table-header
   '("register-finalizer called" "objects" "registration wall ms"
     "registrations/s" "finalizer runs" "never ran" "ran twice"
     "major collections to drain" "drain ms" "exactly once" "load"))
  (print-table-row (cons "plain, first registrations in the process"
                         (finalizer-round 100000 #:protect plain)))
  (print-table-row (cons "plain" (finalizer-round 100000 #:protect plain)))
  (print-table-row (cons "plain, one thread making all of them"
                         (finalizer-round 800000 #:protect plain #:threads 1)))
  (print-table-row (cons "under parameterize-break #f"
                         (finalizer-round 100000 #:protect break-disabled)))
  (print-table-row (cons "in uninterruptible mode"
                         (finalizer-round 100000 #:protect call-as-uninterruptible)))
  (newline)
  (print-table-header
   '("phantoms" "churn (set to the churn size and back)" "expected MiB"
     "held MiB" "after churn MiB" "after zeroing MiB" "set-phantom-bytes!/s"
     "make wall ms" "zero wall ms" "load"))
  (print-table-row (phantom-round 2000 (* 64 1024) 20000 (* 16 mib)))
  (print-table-row (phantom-round 1 (* 64 1024) 100000 (* 64 mib)))
  (print-table-row (phantom-round 1 (* 64 1024) 800000 (* 64 mib) #:threads 1))
  (newline)
  (print-table-header
   '("table, distinct keys per thread" "keys" "count after inserts"
     "wrong or missing values" "count after removes" "correct"
     "insert wall ms" "remove wall ms" "load"))
  (for ([protect (list plain call-as-uninterruptible)]
        [how (list "plain" "uninterruptible")])
    (print-table-row (table-round (format "make-hasheq, ~a" how)
                                  make-hasheq 100000 #:protect protect))
    (print-table-row (table-round (format "make-weak-hasheq, ~a" how)
                                  make-weak-hasheq 100000 #:protect protect))
    (print-table-row (table-round (format "make-late-weak-hasheq (alloc.rkt's), ~a" how)
                                  make-late-weak-hasheq 100000 #:protect protect)))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))

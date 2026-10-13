#lang racket/base

(require (only-in '#%foreign make-late-weak-hasheq)
         (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic call-as-uninterruptible)
         (only-in racket/format ~r)
         (only-in racket/list count)
         (only-in racket/match match-define)
         "common.rkt")

(define default-threads 8)

(define (collect-until done? #:limit-ms [limit-ms 60000])
  (define deadline (+ (now-ms) limit-ms))
  (define collecting
    (measure-span
     (lambda ()
       (define rounds
         (let loop ([rounds 1])
           (collect-garbage 'major)
           (sleep 0.02)
           (cond
             [(or (done?) (> (now-ms) deadline)) rounds]
             [else (loop (add1 rounds))])))
       (define-values (settled settling) (settle-wills!))
       (cons (+ rounds settling) settled))))
  (match-define (cons rounds settled?) (span-value collecting))
  (values rounds (span-ms collecting) settled?))

(define (finalizer-round per-thread #:protect protect #:threads [threads default-threads])
  (define total (* threads per-thread))
  (define runs (build-vector total (lambda (_) (box 0))))
  (define (run-count id) (unbox (vector-ref runs id)))
  (define ms
    (span-ms
     (run-parallel threads
                   (lambda (t)
                     (for ([k (in-range per-thread)])
                       (define id (+ (* t per-thread) k))
                       (protect
                        (lambda ()
                          (register-finalizer (make-vector 2 id)
                                              (lambda (_v)
                                                (cas-update! (vector-ref runs id) add1))))))))))
  (define-values (rounds drain-ms settled?)
    (collect-until (lambda () (for/and ([id (in-range total)]) (positive? (run-count id))))))
  (define final-counts (build-list total run-count))
  (define never (count zero? final-counts))
  (define twice (count (lambda (n) (> n 1)) final-counts))
  (values (list total (real->decimal-string ms 1) (rate->string (/ total (ms->seconds ms)))
                (apply + final-counts) never twice rounds (real->decimal-string drain-ms 1)
                (if (and settled? (zero? never) (zero? twice)) "yes" "NO")
                (load-average))
          (cond
            [(positive? never) (format "~a finalizers never ran" never)]
            [(not settled?) "wills still running after settling"]
            [else #f])))

(define (plain f) (f))
(define (break-disabled f) (parameterize-break #f (f)))

(define (settled-memory)
  (collect-garbage 'major)
  (collect-garbage 'major)
  (current-memory-use))

(define (phantom-round count-per-thread size churn churn-size
                       #:threads [threads default-threads])
  (define base (settled-memory))
  (define made
    (run-parallel threads
                  (lambda (_t)
                    (for/vector #:length count-per-thread
                                ([_ (in-range count-per-thread)])
                      (make-phantom-bytes size)))))
  (define phantoms (list->vector (span-value made)))
  (define held (- (settled-memory) base))
  (define expected (* threads count-per-thread size))
  (define churned
    (run-parallel threads
                  (lambda (t)
                    (define mine (vector-ref phantoms t))
                    (for ([i (in-range churn)])
                      (define p (vector-ref mine (modulo i count-per-thread)))
                      (set-phantom-bytes! p churn-size)
                      (set-phantom-bytes! p size)))))
  (define after-churn (- (settled-memory) base))
  (define zeroed
    (run-parallel threads
                  (lambda (t)
                    (for ([p (in-vector (vector-ref phantoms t))])
                      (set-phantom-bytes! p 0)))))
  (define after-zero (- (settled-memory) base))
  (list (format "~a x ~a x ~a KiB" threads count-per-thread (quotient size 1024))
        (format "~a x ~a MiB" (* threads churn) (~r (bytes->mib churn-size)))
        (bytes->mib-string expected)
        (bytes->mib-string held)
        (bytes->mib-string after-churn)
        (bytes->mib-string after-zero)
        (rate->string (/ (* 2 threads churn) (ms->seconds (span-ms churned))))
        (real->decimal-string (span-ms made) 1)
        (real->decimal-string (span-ms zeroed) 1)
        (load-average)))

(define (table-round label make-table per-thread #:protect protect
                     #:threads [threads default-threads])
  (define table (make-table))
  (define keys
    (for/vector ([_ (in-range (* threads per-thread))]) (make-vector 1 #f)))
  (define (each-key t f)
    (for ([k (in-range per-thread)])
      (define id (+ (* t per-thread) k))
      (protect (lambda () (f id (vector-ref keys id))))))
  (define insert-ms
    (span-ms
     (run-parallel threads
                   (lambda (t)
                     (each-key t (lambda (id key) (hash-set! table key id)))))))
  (define count-after-insert (hash-count table))
  (define mismatched
    (for/sum ([(key id) (in-indexed keys)])
      (if (eqv? (hash-ref table key #f) id) 0 1)))
  (define remove-ms
    (span-ms
     (run-parallel threads
                   (lambda (t)
                     (each-key t (lambda (_id key) (hash-remove! table key)))))))
  (list label
        (* threads per-thread)
        count-after-insert
        mismatched
        (hash-count table)
        (if (and (= count-after-insert (* threads per-thread))
                 (zero? mismatched)
                 (zero? (hash-count table)))
            "yes" "NO")
        (real->decimal-string insert-ms 1)
        (real->decimal-string remove-ms 1)
        (load-average)))

(define (print-finalizer-table)
  (print-table-header
   '("register-finalizer called" "objects" "registration wall ms"
     "registrations/s" "finalizer runs" "never ran" "ran twice"
     "major collections to drain" "drain ms" "exactly once" "load"))
  (for/or ([spec (in-list
                  (list (list "plain, first registrations in the process" 100000 plain 8)
                        (list "plain" 100000 plain 8)
                        (list "plain, one thread making all of them" 800000 plain 1)
                        (list "under parameterize-break #f" 100000 break-disabled 8)
                        (list "in uninterruptible mode" 100000 call-as-uninterruptible 8)))])
    (match-define (list label per-thread protect threads) spec)
    (define-values (row pending) (finalizer-round per-thread #:protect protect #:threads threads))
    (print-table-row (cons label row))
    (and pending (format "~a: ~a" label pending))))

(define (print-phantom-table)
  (print-table-header
   '("phantoms" "churn (set to the churn size and back)" "expected MiB"
     "held MiB" "after churn MiB" "after zeroing MiB" "set-phantom-bytes!/s"
     "make wall ms" "zero wall ms" "load"))
  (print-table-row (phantom-round 2000 (* 64 1024) 20000 (mib->bytes 16)))
  (print-table-row (phantom-round 1 (* 64 1024) 100000 (mib->bytes 64)))
  (print-table-row (phantom-round 1 (* 64 1024) 800000 (mib->bytes 64) #:threads 1)))

(define (print-eq-table)
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
                                  make-late-weak-hasheq 100000 #:protect protect))))

(module+ main
  (print-banner "Q3: finalizers, phantom bytes and eq? tables from 8 parallel threads")
  (define pending (print-finalizer-table))
  (cond
    [pending
     (printf "\nstopped: ~a\n" pending)]
    [else
     (newline)
     (print-phantom-table)
     (newline)
     (print-eq-table)])
  (print-footer))

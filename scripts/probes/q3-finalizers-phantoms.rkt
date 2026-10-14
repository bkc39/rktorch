#lang racket/base

(require (for-syntax racket/base)
         (only-in '#%foreign make-late-weak-hasheq)
         (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic call-as-uninterruptible)
         (only-in racket/format ~r)
         (only-in racket/list count)
         (only-in racket/match match match-define)
         (only-in racket/set seteq set-add set-count set-remove)
         ;; whole-module require on purpose (only-in breaks its expansion)
         racket/runtime-path
         (only-in threading ~>)
         "common.rkt")

(define-runtime-path q3-path "q3-finalizers-phantoms.rkt")

(define default-threads 8)
(define drain-limit-ms 60000)

(define (collect-until done?)
  (define deadline (+ (now-ms) drain-limit-ms))
  (define collecting
    (measure-span
     (lambda ()
       (let loop ([rounds 1])
         (collect-garbage 'major)
         (sleep 0.02)
         (cond
           [(done?)
            (define-values (settled settling) (settle-wills!))
            (list (+ rounds settling) settled #f)]
           [(> (now-ms) deadline) (list rounds #f #t)]
           [else (loop (add1 rounds))])))))
  (match-define (list rounds settled? timed-out?) (span-value collecting))
  (values rounds (span-ms collecting) settled? timed-out?))

(define (finalizer-round per-thread #:protect protect #:threads [threads default-threads])
  (define total (* threads per-thread))
  (define runs (build-vector total (lambda (_) (box 0))))
  (define ran-on (box (seteq)))
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
                                                (with-finalizer-in-flight
                                                  (cas-update! ran-on set-add (current-thread))
                                                  (cas-update! (vector-ref runs id) add1)))))))))))
  (define-values (rounds drain-ms settled? timed-out?)
    (collect-until (lambda () (for/and ([id (in-range total)]) (positive? (run-count id))))))
  (define final-counts (build-list total run-count))
  (define never (count zero? final-counts))
  (define twice (count (lambda (n) (> n 1)) final-counts))
  (define finalizer-threads (~> (unbox ran-on) (set-remove (current-thread)) set-count))
  (values (list total (real->decimal-string ms 1) (rate->string (/ total (ms->seconds ms)))
                (apply + final-counts) finalizer-threads never twice rounds
                (real->decimal-string drain-ms 1)
                (if (and settled? (zero? never) (zero? twice)) "yes" "NO")
                (load-average))
          (cond
            [timed-out? (format "finalizers still pending after ~a s of collections"
                                (ms->seconds drain-limit-ms))]
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

(define (start-finalizer-thread-on-main!)
  (settle-wills!)
  (void))

(struct finalizer-spec (label per-thread protect threads prepare))

(define finalizer-specs
  (vector (finalizer-spec "plain, first registrations in the process" 100000 plain 8 void)
          (finalizer-spec "plain" 100000 plain 8 start-finalizer-thread-on-main!)
          (finalizer-spec "plain, one thread making all of them" 800000 plain 1
                          start-finalizer-thread-on-main!)
          (finalizer-spec "under parameterize-break #f" 100000 break-disabled 8
                          start-finalizer-thread-on-main!)
          (finalizer-spec "in uninterruptible mode" 100000 call-as-uninterruptible 8
                          start-finalizer-thread-on-main!)))

(struct finalizer-outcome (raised-wills) #:prefab)
(struct finalizer-measured finalizer-outcome (row pending) #:prefab)
(struct finalizer-unsettled finalizer-outcome (reason) #:prefab)

(define (finalizer-row i)
  (match-define (finalizer-spec _ per-thread protect threads prepare)
    (vector-ref finalizer-specs i))
  (with-handlers ([exn:fail:unsettled?
                   (lambda (e) (finalizer-unsettled (raised-wills) (exn-message e)))])
    (prepare)
    (define-values (row pending) (finalizer-round per-thread #:protect protect #:threads threads))
    (finalizer-measured (raised-wills) row pending)))

(module* finalizer-row #f
  (provide finalizer-row))

(define (finalizer-row-in-fresh-racket i)
  (run-fresh-racket
   `(begin
      (require (submod (file ,(path->string q3-path)) finalizer-row))
      (write (finalizer-row ,i)))))

(define (print-finalizer-table)
  (print-table-header
   '("register-finalizer called (fresh process per row)" "objects" "registration wall ms"
     "registrations/s" "finalizer runs" "finalizer threads running them" "never ran" "ran twice"
     "major collections to drain" "drain ms" "exactly once" "load"))
  (for/or ([spec (in-vector finalizer-specs)] [i (in-naturals)])
    (define label (finalizer-spec-label spec))
    (with-unsettled-stop label
      (define outcome (finalizer-row-in-fresh-racket i))
      (record-raised-wills! (finalizer-outcome-raised-wills outcome))
      (match outcome
        [(finalizer-unsettled _ reason) reason]
        [(finalizer-measured _ row pending)
         (print-table-row (cons label row))
         pending]))))

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
  (for* ([protect (list plain call-as-uninterruptible)]
         [make-table (list make-hasheq make-weak-hasheq make-late-weak-hasheq)])
    (~> (format "~a, ~a" (object-name make-table) (object-name protect))
        (table-round make-table 100000 #:protect protect)
        print-table-row)))

(module+ main
  (print-banner "Q3: finalizers, phantom bytes and eq? tables from 8 parallel threads")
  (define stopped (print-finalizer-table))
  (unless stopped
    (newline)
    (print-phantom-table)
    (newline)
    (print-eq-table))
  (print-footer #:stopped stopped))

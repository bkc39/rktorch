#lang racket/base

(require (only-in racket/math exact-floor))

(provide margin-over
         memory-fraction-from-env
         memory-limit-from-env
         native-collect-at-troughs
         native-collect-budget
         native-collect-margin
         native-memory-fraction
         native-memory-limit
         release-spacing)

(define mib (* 1024 1024))

(define (setting-from-env name parse default expected)
  (define supplied (getenv name))
  (cond
    [(not supplied) default]
    [(parse (string->number supplied 10 'number-or-false 'decimal-as-exact))]
    [else
     (raise-user-error (string->symbol name) "expected ~a, given ~s"
                       expected supplied)]))

(define (fraction n)
  (and (real? n) (< 0 n) (<= n 1) n))

(define (mib->bytes n)
  (and (rational? n) (positive? n)
       (let ([bytes (exact-floor (* n mib))])
         (and (positive? bytes) bytes))))

(define (memory-limit-from-env)
  (setting-from-env "RKTORCH_MEMORY_LIMIT" mib->bytes #f "a positive number of MiB"))

(define (memory-fraction-from-env)
  (setting-from-env "RKTORCH_MEMORY_FRACTION" fraction 4/5 "a number in (0, 1]"))

(define native-memory-limit (make-parameter (memory-limit-from-env)))
(define native-memory-fraction (make-parameter (memory-fraction-from-env)))

(define release-spacing (make-parameter 5000.0))

;; #f: the floor itself, kept between the two bounds below
(define native-collect-margin (make-parameter #f))
(define margin-min (* 256 mib))
(define margin-max (* 1024 mib))

(define (margin-over floor)
  (or (native-collect-margin)
      (max margin-min (min floor margin-max))))

;; the share of wall-clock time trough collections may take
(define native-collect-budget (make-parameter 1/20))

;; whether backward! and an outermost no-grad layer call collect at all
(define native-collect-at-troughs (make-parameter #t))

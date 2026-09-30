#lang racket/base

(require (only-in ffi/unsafe _bytes _fun _int _intptr _void get-ffi-obj))

(provide latched
         native-fault?
         native-fault-policy
         native-faulted
         note-native-fault!)

(define (native-fault? v)
  (and (exn:fail? v)
       (regexp-match? #rx"^invalid memory reference" (exn-message v))))

(define native-fault-policy
  (if (equal? (getenv "RKTORCH_ON_NATIVE_FAULT") "exit") 'exit 'raise))

;; write(2) and _exit(2), not ports and `exit`: this runs from finalizers in
;; atomic mode, and after a fault nothing above libc can be trusted to return.
(define c-write (get-ffi-obj "write" #f (_fun _int _bytes _intptr -> _intptr)))
(define c-exit (get-ffi-obj "_exit" #f (_fun _int -> _void)))

(define exit-code 70)

(define (tell! . parts)
  (define message (string->bytes/utf-8 (apply string-append parts)))
  (c-write 2 message (bytes-length message)))

(define faulted (box #f))

(define (native-faulted) (unbox faulted))

(define (note-native-fault! doing)
  (case native-fault-policy
    [(exit)
     (tell! "rktorch: native code faulted (invalid memory reference) while "
            doing ".\nThe native heap may be corrupt, so the process is exiting.\n")
     (c-exit exit-code)]
    [else
     (unless (unbox faulted)
       (set-box! faulted doing)
       (tell! "rktorch: native code faulted (invalid memory reference) while "
              doing ".\nThe native library is disabled for the rest of this"
              " process.\n"))]))

(define (raise-faulted)
  (raise
   (exn:fail
    (string-append
     "rktorch: native code faulted earlier, while " (unbox faulted)
     ", so the native library is disabled for the rest of this process;"
     " restart it.\n  Set RKTORCH_ON_NATIVE_FAULT=exit to stop the process"
     " at the first fault instead.")
    (current-continuation-marks))))

(define (latched f)
  (case (procedure-arity f)
    [(0) (lambda () (when (unbox faulted) (raise-faulted)) (f))]
    [(1) (lambda (a) (when (unbox faulted) (raise-faulted)) (f a))]
    [(2) (lambda (a b) (when (unbox faulted) (raise-faulted)) (f a b))]
    [(3) (lambda (a b c) (when (unbox faulted) (raise-faulted)) (f a b c))]
    [(4) (lambda (a b c d) (when (unbox faulted) (raise-faulted)) (f a b c d))]
    [(5) (lambda (a b c d e)
           (when (unbox faulted) (raise-faulted))
           (f a b c d e))]
    [(6) (lambda (a b c d e g)
           (when (unbox faulted) (raise-faulted))
           (f a b c d e g))]
    [else (lambda args (when (unbox faulted) (raise-faulted)) (apply f args))]))

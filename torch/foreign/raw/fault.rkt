#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe _bytes _fun _int _intptr _void get-ffi-obj)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide crossing-hook
         latched
         native-fault?
         native-fault-policy
         native-faulted
         note-native-fault!
         reset-native-fault-latch!)

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

(define (reset-native-fault-latch!) (set-box! faulted #f))

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

(define crossing-hook (box #f))

(define-syntax-parse-rule (enter!)
  (begin
    (when (unbox faulted) (raise-faulted))
    (let ([hook (unbox crossing-hook)])
      (when hook (hook)))))

(define (latched f)
  (case (procedure-arity f)
    [(0) (lambda () (enter!) (f))]
    [(1) (lambda (a) (enter!) (f a))]
    [(2) (lambda (a b) (enter!) (f a b))]
    [(3) (lambda (a b c) (enter!) (f a b c))]
    [(4) (lambda (a b c d) (enter!) (f a b c d))]
    [(5) (lambda (a b c d e) (enter!) (f a b c d e))]
    [(6) (lambda (a b c d e g) (enter!) (f a b c d e g))]
    [else (lambda args (enter!) (apply f args))]))

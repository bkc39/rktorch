#lang racket/base

(require (only-in ffi/unsafe _bytes _fun _int _intptr _uint get-ffi-obj)
         (only-in ffi/unsafe/os-thread call-in-os-thread)
         (only-in racket/port port->string))

(provide arm-process-deadline!
         call-with-process-deadline
         exit-status-of)

(define c-write (get-ffi-obj "write" #f (_fun _int _bytes _intptr -> _intptr)))
(define c-sleep (get-ffi-obj "sleep" #f (_fun #:blocking? #t _uint -> _uint)))
(define c-raise (get-ffi-obj "raise" #f (_fun _int -> _int)))
(define sigalrm 14)

;; The watch runs on an OS thread of its own, so it ends a process wedged in
;; atomic mode, and it names itself on stderr first: under raco cover the
;; process it ends is the whole suite's. It sleeps to the deadline in one
;; blocking call, and a lifted watch sleeps on: a thread that returns from a
;; blocking call after a collection claims a fresh 16 KiB allocation area,
;; which (current-memory-use 'cumulative) counts in whatever window main has
;; open, so a watch that woke to poll would land in heap measurements.
(define (arm-process-deadline! seconds [what "a deadline's body"])
  (define lifted (box #f))
  (define notice
    (string->bytes/utf-8
     (format "process deadline: ~a ran past ~a s; ending the process with SIGALRM\n"
             what seconds)))
  (call-in-os-thread
   (lambda ()
     (let watch ([left (inexact->exact (ceiling seconds))])
       (define unslept (c-sleep left))
       (cond
         [(unbox lifted) (void)]
         [(positive? unslept) (watch unslept)]
         [else
          (c-write 2 notice (bytes-length notice))
          (c-raise sigalrm)]))))
  (lambda () (set-box! lifted #t)))

(define (call-with-process-deadline seconds thunk)
  (define lift! #f)
  (dynamic-wind (lambda ()
                  (set! lift! (arm-process-deadline! seconds (or (object-name thunk) "a thunk"))))
                thunk
                (lambda () (lift!))))

(define (status-and-stderr-of program #:within seconds)
  (define-values (sp out in err)
    (subprocess #f #f #f (find-system-path 'exec-file) "-e" (format "~s" program)))
  (close-output-port in)
  (define stderr (box ""))
  (define drains
    (list (thread (lambda () (port->string out)))
          (thread (lambda () (set-box! stderr (port->string err))))))
  (define status
    (cond
      [(sync/timeout seconds sp) (subprocess-status sp)]
      [else (subprocess-kill sp #t) #f]))
  (for ([t (in-list drains)]) (sync/timeout 10 t))
  (close-input-port out)
  (close-input-port err)
  (values status (unbox stderr)))

(define (exit-status-of program #:within seconds)
  (define-values (status _stderr) (status-and-stderr-of program #:within seconds))
  status)

(module+ test
  (require rackunit)

  (define sigalrm-status (+ 128 14))

  (define wedge-module (path->string (collection-file-path "wedge.rkt" "torch" "tests" "private")))

  (define (program body)
    `(begin
       (module run racket/base
         (require (only-in ffi/unsafe _fun _int _uint get-ffi-obj)
                  (only-in ffi/unsafe/atomic start-atomic)
                  (file ,wedge-module))
         (define usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))
         (define (wedge) ;; noqa
           (start-atomic)
           (let loop ()
             (usleep 100000)
             (loop)))
         ,body)
       (require 'run)))

  (define (run body #:within seconds)
    (exit-status-of (program body) #:within seconds))

  (test-case "a process wedged in atomic mode ends at its deadline"
    (check-equal? (run '(call-with-process-deadline 1 wedge) #:within 30) sigalrm-status))

  (test-case "a deadline that ends the process names its body on stderr first"
    (define-values (status stderr)
      (status-and-stderr-of (program '(call-with-process-deadline 1 wedge)) #:within 30))
    (check-equal? status sigalrm-status)
    (check-regexp-match #rx"process deadline: wedge ran past 1 s" stderr))

  (test-case "the deadline is lifted when its body returns"
    (check-equal? (run '(begin (call-with-process-deadline 1 void) (sleep 2)) #:within 30) 0))

  (test-case "the deadline is lifted when its body raises"
    (check-equal? (run '(begin
                          (with-handlers ([(lambda (_v) #t) void])
                            (call-with-process-deadline 1 (lambda () (raise 'escaped))))
                          (sleep 2))
                       #:within 30)
                  0))

  (test-case "an inner deadline's end re-arms the outer one"
    (check-equal? (run '(call-with-process-deadline
                         2
                         (lambda ()
                           (call-with-process-deadline 1 void)
                           (wedge)))
                       #:within 10)
                  sigalrm-status))

  (test-case "an inner deadline does not extend the outer one"
    (check-equal? (run '(call-with-process-deadline
                         1
                         (lambda () (call-with-process-deadline 60 wedge)))
                       #:within 10)
                  sigalrm-status))

  (test-case "an armed deadline adds nothing to main's heap allocation (#460)"
    (define-values (status stderr)
      (status-and-stderr-of
       (program '(begin
                   (arm-process-deadline! 300)
                   (usleep 200000)
                   (define spans
                     (for/list ([_ (in-range 3)])
                       (collect-garbage 'minor)
                       (define start (current-memory-use 'cumulative))
                       (usleep 200000)
                       (- (current-memory-use 'cumulative) start)))
                   (eprintf "~a" (apply max spans))))
       #:within 30))
    (check-equal? status 0 stderr)
    (define bytes (string->number stderr))
    (check-true (and bytes (< bytes 4096))
                (format "~a bytes in 200 ms of main's sleep" stderr))))

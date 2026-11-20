#lang racket/base

(require (only-in ffi/unsafe _bytes _fun _int _intptr _uint get-ffi-obj)
         (only-in ffi/unsafe/os-thread call-in-os-thread)
         (only-in racket/match match)
         (only-in racket/os getpid)
         (only-in racket/port port->string))

(provide arm-process-deadline!
         call-with-process-deadline
         deadline-scale
         deadline-scale-variable
         deadlines-armed
         exit-status-of
         status-and-stderr-of)

(define c-write (get-ffi-obj "write" #f (_fun _int _bytes _intptr -> _intptr)))
(define c-sleep (get-ffi-obj "sleep" #f (_fun #:blocking? #t _uint -> _uint)))
(define c-raise (get-ffi-obj "raise" #f (_fun _int -> _int)))
(define sigalrm 14)

(define-logger process-deadline)

(define deadline-scale-variable "RKTORCH_DEADLINE_SCALE")

(define (deadline-scale)
  (match (getenv deadline-scale-variable)
    [(pregexp #px"^([1-9][0-9]*)@([0-9]+)$" (list _ factor pid))
     #:when (= (string->number pid) (getpid))
     (string->number factor)]
    [_ 1]))

(define (bound-text seconds scale)
  (match scale
    [1 (format "~a s" seconds)]
    [_ (format "~a s (~a s x ~a under raco cover)" (* scale seconds) seconds scale)]))

;; The watch runs on an OS thread of its own, so it ends a process wedged in
;; atomic mode, and it names itself on stderr first: under raco cover the
;; process it ends is the whole suite's, which runs instrumented, so there the
;; deadline is scaled by the factor cover stamped with that process's pid; its
;; children inherit the stamp but load compiled code, and keep theirs. It
;; sleeps to the deadline in one blocking call, and a lifted watch sleeps on: a
;; thread that returns from a blocking call after a collection claims a fresh
;; 16 KiB allocation area, which (current-memory-use 'cumulative) counts in
;; whatever window main has open, so a watch that woke to poll would land in
;; heap measurements.
(define (arm-process-deadline! seconds [what "a deadline's body"])
  (define scale (deadline-scale))
  (define lifted (box #f))
  (define notice
    (string->bytes/utf-8
     (format "process deadline: ~a ran past ~a; ending the process with SIGALRM\n"
             what (bound-text seconds scale))))
  (log-process-deadline-debug "armed for ~a: ~a" what (bound-text seconds scale))
  (call-in-os-thread
   (lambda ()
     (let watch ([left (inexact->exact (ceiling (* scale seconds)))])
       (define unslept (c-sleep left))
       (cond
         [(unbox lifted) (void)]
         [(positive? unslept) (watch unslept)]
         [else
          (c-write 2 notice (bytes-length notice))
          (c-raise sigalrm)]))))
  (lambda () (set-box! lifted #t)))

(define (call-with-process-deadline seconds thunk
                                    #:what [what (or (object-name thunk) "a thunk")])
  (define lift! #f)
  (dynamic-wind (lambda ()
                  (set! lift! (arm-process-deadline! seconds what)))
                thunk
                (lambda () (lift!))))

(define (deadlines-armed thunk)
  (define receiver (make-log-receiver (current-logger) 'debug 'process-deadline))
  (thunk)
  (for/sum ([_ (in-producer (lambda () (sync/timeout 0 receiver)) #f)]) 1))

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
                  (only-in racket/os getpid)
                  (file ,wedge-module))
         (define usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))
         (define (stamp! factor pid) ;; noqa
           (putenv deadline-scale-variable (format "~a@~a" factor pid)))
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

  (test-case "a deadline names the body it was given a name for"
    (define-values (status stderr)
      (status-and-stderr-of (program '(call-with-process-deadline 1 wedge #:what "the unwind"))
                            #:within 30))
    (check-equal? status sigalrm-status)
    (check-regexp-match #rx"process deadline: the unwind ran past 1 s" stderr))

  (test-case "in the process raco cover stamped, a deadline is scaled by the stamp's factor"
    (check-equal? (run '(begin
                          (stamp! 8 (getpid))
                          (call-with-process-deadline 1 (lambda () (sleep 3))))
                       #:within 30)
                  0))

  (test-case "a scaled deadline that ends the process names both bounds"
    (define-values (status stderr)
      (status-and-stderr-of
       (program '(begin (stamp! 2 (getpid)) (call-with-process-deadline 1 wedge)))
       #:within 30))
    (check-equal? status sigalrm-status)
    (check-regexp-match #rx"wedge ran past 2 s [(]1 s x 2 under raco cover[)]" stderr))

  (test-case "a child that inherits another process's stamp keeps its own deadline"
    (check-equal? (run `(begin (stamp! 8 ,(getpid)) (call-with-process-deadline 1 wedge))
                       #:within 30)
                  sigalrm-status))

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

#lang racket/base

(require (only-in ffi/unsafe
                  _bytes
                  _fun
                  _int
                  _intptr
                  _pointer
                  get-ffi-obj
                  malloc)
         (only-in ffi/unsafe/os-thread call-in-os-thread)
         (only-in racket/match match match-define)
         (only-in racket/math exact-floor)
         (only-in racket/os getpid)
         (only-in racket/port port->string)
         (only-in "clock.rkt"
                  clock-gettime
                  make-timespec
                  set-timespec-nsec!
                  set-timespec-sec!
                  timespec-nsec
                  timespec-sec)
         (only-in "platform.rkt" host-platform))

(provide arm-process-deadline!
         call-with-process-deadline
         deadline-scale
         deadline-scale-variable
         deadlines-armed
         exit-status-of
         status-and-stderr-of)

(define (libc name type) (get-ffi-obj name #f type))
(define c-write (libc "write" (_fun _int _bytes _intptr -> _intptr)))
(define c-raise (libc "raise" (_fun _int -> _int)))
(define sigalrm 14)

(define (pthread-constants os)
  (match os
    ['linux '(110 1)]
    ['macosx '(60 6)]
    [_ (raise-arguments-error 'process-deadline "no ETIMEDOUT known for this OS" "os" os)]))

(match-define (list etimedout clock-monotonic) (pthread-constants host-platform))

(define (call-failed! call status)
  (define notice
    (string->bytes/utf-8
     (string-append "process deadline: " call " returned " (number->string status)
                    "; ending the process with SIGALRM\n")))
  (c-write 2 notice (bytes-length notice))
  (c-raise sigalrm))

(define ((checked name call) . arguments)
  (define status (apply call arguments))
  (unless (zero? status)
    (call-failed! name status)))

(define (checked-call name type)
  (checked name (libc name type)))

(define checked-read-clock (checked "clock_gettime" clock-gettime))
(define mutex-init (checked-call "pthread_mutex_init" (_fun _pointer _pointer -> _int)))
(define mutex-lock (checked-call "pthread_mutex_lock" (_fun #:blocking? #t _pointer -> _int)))
(define mutex-unlock (checked-call "pthread_mutex_unlock" (_fun _pointer -> _int)))
(define condition-init (checked-call "pthread_cond_init" (_fun _pointer _pointer -> _int)))
(define condition-signal (checked-call "pthread_cond_signal" (_fun _pointer -> _int)))
(define condition-wait
  (checked-call "pthread_cond_wait" (_fun #:blocking? #t _pointer _pointer -> _int)))

(define pthread-object-bytes 128)

(define (pthread-object init! . arguments)
  (define object (malloc pthread-object-bytes 'atomic-interior))
  (apply init! object arguments)
  object)

(define nanoseconds-per-second 1000000000)

(define (timespec->nanoseconds time)
  (+ (* nanoseconds-per-second (timespec-sec time)) (timespec-nsec time)))

(define (set-timespec-nanoseconds! time nanoseconds)
  (define-values (sec nsec) (quotient/remainder nanoseconds nanoseconds-per-second))
  (set-timespec-sec! time sec)
  (set-timespec-nsec! time nsec))

(define (set-time-left! left deadline now)
  (set-timespec-nanoseconds!
   left
   (max 1 (- (timespec->nanoseconds deadline) (timespec->nanoseconds now)))))

(define (monotonic-deadline-after milliseconds)
  (define deadline (make-timespec 0 0))
  (checked-read-clock clock-monotonic deadline)
  (set-timespec-nanoseconds!
   deadline
   (+ (timespec->nanoseconds deadline) (exact-floor (* 1000000 milliseconds))))
  deadline)

(define-values (monotonic-condition-attributes wait-name wait-until)
  (match host-platform
    ['linux
     (define attributes-init (checked-call "pthread_condattr_init" (_fun _pointer -> _int)))
     (define attributes-set-clock
       (checked-call "pthread_condattr_setclock" (_fun _pointer _int -> _int)))
     (define wait-until-deadline
       (libc "pthread_cond_timedwait" (_fun #:blocking? #t _pointer _pointer _pointer -> _int)))
     (values (lambda ()
               (define attributes (pthread-object attributes-init))
               (attributes-set-clock attributes clock-monotonic)
               attributes)
             "pthread_cond_timedwait"
             (lambda (condition mutex deadline _left)
               (wait-until-deadline condition mutex deadline)))]
    ['macosx
     (define wait-for
       (libc "pthread_cond_timedwait_relative_np"
             (_fun #:blocking? #t _pointer _pointer _pointer -> _int)))
     (values (lambda () #f)
             "pthread_cond_timedwait_relative_np"
             (lambda (condition mutex deadline left)
               (clock-gettime clock-monotonic left)
               (set-time-left! left deadline left)
               (wait-for condition mutex left)))]))

(define (wake-outcome status lifted?)
  (cond
    [lifted? 'lifted]
    [(zero? status) 'spurious]
    [(= status etimedout) 'ran-past]
    [else 'failed]))

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

;; arm returns only once the watch is parked in its blocking wait, so the watch
;; charges no allocation to any heap window main opens afterwards (#460).
(define (arm-process-deadline! seconds [what "a deadline's body"])
  (define scale (deadline-scale))
  (define bound (bound-text seconds scale))
  (define lifted (box #f))
  (define parked (box #f))
  (define mutex (pthread-object mutex-init #f))
  (define condition (pthread-object condition-init (monotonic-condition-attributes)))
  (define deadline (monotonic-deadline-after (* 1000 scale seconds)))
  (define left (make-timespec 0 0))
  (define notice
    (string->bytes/utf-8
     (format "process deadline: ~a ran past ~a; ending the process with SIGALRM\n" what bound)))
  (define notice-length (bytes-length notice))
  (log-process-deadline-debug "armed for ~a: ~a" what bound)
  (mutex-lock mutex)
  (call-in-os-thread
   (lambda ()
     (mutex-lock mutex)
     (set-box! parked #t)
     (condition-signal condition)
     (let watch ()
       (define status (wait-until condition mutex deadline left))
       (match (wake-outcome status (unbox lifted))
         ['lifted (void)]
         ['spurious (watch)]
         ['ran-past
          (c-write 2 notice notice-length)
          (c-raise sigalrm)]
         ['failed (call-failed! wait-name status)]))))
  (let park ()
    (unless (unbox parked)
      (condition-wait condition mutex)
      (park)))
  (mutex-unlock mutex)
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

(define (status-and-stderr-of program #:within seconds #:rerun-unless [notice #f])
  (define-values (status stderr) (status-and-stderr-once program seconds))
  (cond
    [(and notice (not (regexp-match? notice stderr))) (status-and-stderr-once program seconds)]
    [else (values status stderr)]))

(define (status-and-stderr-once program seconds)
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
  (require (only-in racket/file make-temporary-file)
           rackunit)

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

  (test-case "ETIMEDOUT and CLOCK_MONOTONIC are read for the OS os* names, and others refused"
    (check-equal? (pthread-constants 'linux) '(110 1))
    (check-equal? (pthread-constants 'macosx) '(60 6))
    (check-exn #rx"no ETIMEDOUT known for this OS.*freebsd"
               (lambda () (pthread-constants 'freebsd))))

  (test-case "a wake is read as lifted first, then as spurious, a deadline, or a failed call"
    (check-equal? (wake-outcome 22 #t) 'lifted)
    (check-equal? (wake-outcome etimedout #t) 'lifted)
    (check-equal? (wake-outcome 0 #f) 'spurious)
    (check-equal? (wake-outcome etimedout #f) 'ran-past)
    (check-equal? (wake-outcome 22 #f) 'failed))

  (test-case "the time left to a deadline is its difference from now, and never zero"
    (define (left-of deadline now)
      (define left (make-timespec 0 0))
      (set-time-left! left (apply make-timespec deadline) (apply make-timespec now))
      (list (timespec-sec left) (timespec-nsec left)))
    (check-equal? (left-of '(10 500000000) '(8 700000000)) '(1 800000000))
    (check-equal? (left-of '(10 0) '(10 0)) '(0 1))
    (check-equal? (left-of '(10 0) '(12 0)) '(0 1)))

  (test-case "a wedge ends at its bound, not before it"
    (define start (current-inexact-monotonic-milliseconds))
    (check-equal? (run '(call-with-process-deadline 2 wedge) #:within 30) sigalrm-status)
    (define elapsed (- (current-inexact-monotonic-milliseconds) start))
    (check-true (>= elapsed 2000) (format "ended after ~a ms" elapsed)))

  (test-case "a process wedged in atomic mode ends at its deadline"
    (check-equal? (run '(call-with-process-deadline 1 wedge) #:within 30) sigalrm-status))

  (test-case "a deadline that ends the process names its body on stderr first"
    (define-values (status stderr)
      (status-and-stderr-of (program '(call-with-process-deadline 1 wedge)) #:within 30))
    (check-equal? status sigalrm-status)
    (check-regexp-match #rx"process deadline: wedge ran past 1 s" stderr))

  (test-case "a run without the notice is rerun once, and a run with it is not"
    (define marker (make-temporary-file "wedge-rerun~a"))
    (define (outcome first-run)
      (delete-file marker)
      (define-values (status stderr)
        (status-and-stderr-of
         `(cond
            [(file-exists? ,(path->string marker)) (exit 3)]
            [else (close-output-port (open-output-file ,(path->string marker))) ,first-run])
         #:within 30
         #:rerun-unless #rx"process deadline: "))
      (list status stderr))
    (check-equal? (outcome '(exit 0)) '(3 ""))
    (check-equal? (outcome '(begin (eprintf "process deadline: x") (exit 142)))
                  '(142 "process deadline: x"))
    (delete-file marker))

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

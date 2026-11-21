#lang racket/base

(require (only-in ffi/unsafe
                  _bytes
                  _fun
                  _int
                  _intptr
                  _long
                  _pointer
                  define-cstruct
                  get-ffi-obj
                  malloc)
         (only-in ffi/unsafe/os-thread call-in-os-thread)
         (only-in racket/match match)
         (only-in racket/math exact-floor)
         (only-in racket/os getpid)
         (only-in racket/port port->string))

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
(define etimedout (match (system-type 'os) ['macosx 60] [_ 110]))

(define (pthread-failed! call status)
  (define notice
    (string->bytes/utf-8
     (string-append "process deadline: " call " returned " (number->string status)
                    "; ending the process with SIGALRM\n")))
  (c-write 2 notice (bytes-length notice))
  (c-raise sigalrm))

(define (pthread-call name type)
  (define call (libc name type))
  (lambda arguments
    (define status (apply call arguments))
    (unless (zero? status)
      (pthread-failed! name status))))

(define mutex-init (pthread-call "pthread_mutex_init" (_fun _pointer _pointer -> _int)))
(define mutex-lock (pthread-call "pthread_mutex_lock" (_fun #:blocking? #t _pointer -> _int)))
(define mutex-unlock (pthread-call "pthread_mutex_unlock" (_fun _pointer -> _int)))
(define condition-init (pthread-call "pthread_cond_init" (_fun _pointer _pointer -> _int)))
(define condition-signal (pthread-call "pthread_cond_signal" (_fun _pointer -> _int)))
(define condition-wait
  (pthread-call "pthread_cond_wait" (_fun #:blocking? #t _pointer _pointer -> _int)))
(define condition-wait-until
  (libc "pthread_cond_timedwait" (_fun #:blocking? #t _pointer _pointer _pointer -> _int)))

(define pthread-object-bytes 128)

(define (pthread-object init!)
  (define object (malloc pthread-object-bytes 'atomic-interior))
  (init! object #f)
  object)

(define-cstruct _timespec ([sec _long] [nsec _long]) #:malloc-mode 'atomic-interior)

(define (set-timespec-after! deadline wall-clock milliseconds)
  (define-values (sec ms) (quotient/remainder (exact-floor (+ (wall-clock) milliseconds)) 1000))
  (set-timespec-sec! deadline sec)
  (set-timespec-nsec! deadline (* ms 1000000)))

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
;; children inherit the stamp but load compiled code, and keep theirs. An OS
;; thread that runs at all after a collection is charged a fresh 16 KiB
;; allocation area when it next enters a blocking call, and
;; (current-memory-use 'cumulative) counts it in whatever window main has open.
;; So the watch waits out the deadline in one blocking pthread_cond_timedwait,
;; a lifted watch waits on, and arm returns only once the watch is inside that
;; wait: main gets the mutex back only when the wait releases it, so the watch
;; runs no Racket code in any window main opens after arm returns until the
;; wait wakes, at its deadline or spuriously. The wait's deadline is on the
;; wall clock, so a wake that finds the monotonic target still ahead, as after
;; a clock step, waits out what is left of it.
(define (arm-process-deadline! seconds
                               [what "a deadline's body"]
                               #:wall-clock [wall-clock current-inexact-milliseconds])
  (define scale (deadline-scale))
  (define bound (bound-text seconds scale))
  (define lifted (box #f))
  (define parked (box #f))
  (define mutex (pthread-object mutex-init))
  (define condition (pthread-object condition-init))
  (define milliseconds (* 1000 scale seconds))
  (define target (+ (current-inexact-monotonic-milliseconds) milliseconds))
  (define deadline (make-timespec 0 0))
  (set-timespec-after! deadline wall-clock milliseconds)
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
       (define status (condition-wait-until condition mutex deadline))
       (define remaining (- target (current-inexact-monotonic-milliseconds)))
       (cond
         [(unbox lifted) (void)]
         [(and (= status etimedout) (positive? remaining))
          (set-timespec-after! deadline wall-clock remaining)
          (watch)]
         [(zero? status) (watch)]
         [(= status etimedout)
          (c-write 2 notice notice-length)
          (c-raise sigalrm)]
         [else (pthread-failed! "pthread_cond_timedwait" status)]))))
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
         (define (stepped-forward-by milliseconds) ;; noqa
           (define step (box milliseconds))
           (lambda ()
             (begin0 (- (current-inexact-milliseconds) (unbox step))
                     (set-box! step 0))))
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

  (test-case "a wall clock stepped forward after arming does not end the process early"
    (check-equal? (run '(begin
                          (arm-process-deadline! 3 #:wall-clock (stepped-forward-by 2500))
                          (sleep 1.5))
                       #:within 30)
                  0))

  (test-case "a wall clock stepped forward after arming still ends a wedge at its deadline"
    (define-values (status stderr)
      (status-and-stderr-of
       (program '(begin
                   (arm-process-deadline! 3
                                          "the stepped wedge"
                                          #:wall-clock (stepped-forward-by 2500))
                   (wedge)))
       #:within 30))
    (check-equal? status sigalrm-status)
    (check-regexp-match #rx"process deadline: the stepped wedge ran past 3 s" stderr))

  (test-case "a wait the OS refuses ends the process naming the call, not a deadline"
    (define-values (status stderr)
      (status-and-stderr-of
       (program '(begin
                   (arm-process-deadline! 60 #:wall-clock (lambda () -10000000.5))
                   (sleep 5)))
       #:within 30))
    (check-equal? status sigalrm-status)
    (check-regexp-match #rx"process deadline: pthread_cond_timedwait returned 22;" stderr)
    (check-false (regexp-match? #rx"ran past" stderr) stderr))

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

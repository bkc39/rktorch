#lang racket/base

(require (only-in ffi/unsafe _fun _uint get-ffi-obj)
         (only-in racket/port port->string)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define)

(provide call-with-process-deadline
         exit-status-of
         shut-down-custodian!
         with-process-deadline)

;; Racket 9.3's custodian-shutdown-all holds the custodian lock while it
;; waits, asleep in atomic mode, for a running parallel thread to stop, and
;; the thread needs that lock to stop: the place wedges. raco test's --timeout
;; is a sync/timeout in that same scheduler and never fires. SIGALRM's default
;; action ends the process from the kernel instead; one alarm per process, so
;; an inner deadline replaces an outer one.
(define alarm (get-ffi-obj "alarm" #f (_fun _uint -> _uint)))

(define (call-with-process-deadline seconds thunk)
  (dynamic-wind (lambda () (alarm seconds))
                thunk
                (lambda () (alarm 0))))

(define-syntax-parse-rule (with-process-deadline seconds:expr body:expr ...+) ;; noqa
  (call-with-process-deadline seconds (lambda () body ...))) ;; noqa

;; kill-thread takes no custodian lock, so a running parallel thread can stop
(define (shut-down-custodian! c)
  (for ([v (in-list (custodian-managed-list c (current-custodian)))]
        #:when (thread? v))
    (kill-thread v))
  (custodian-shutdown-all c))

;; #f when the program is still running after `seconds`
(define (exit-status-of program #:within seconds)
  (define-values (sp out in err)
    (subprocess #f #f #f (find-system-path 'exec-file) "-e" (format "~s" program)))
  (close-output-port in)
  (define drains (for/list ([p (list out err)]) (thread (lambda () (port->string p)))))
  (define exited? (and (sync/timeout seconds sp) #t))
  (unless exited? (subprocess-kill sp #t))
  (for ([t (in-list drains)]) (sync/timeout 10 t))
  (close-input-port out)
  (close-input-port err)
  (and exited? (subprocess-status sp)))

(module+ test
  (require rackunit)

  (define sigalrm-status (+ 128 14))

  (define wedge-module (path->string (collection-file-path "wedge.rkt" "torch" "tests" "private")))

  (define (run body #:within seconds)
    (exit-status-of
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
          (define (busy-worker-under c) ;; noqa
            (parameterize ([current-custodian c])
              (thread (lambda () (let loop () (make-vector 1000) (loop))) #:pool 'own)))
          ,body)
        (require 'run))
     #:within seconds))

  (test-case "a process wedged in atomic mode ends at its deadline"
    (check-equal? (run '(with-process-deadline 1 (wedge)) #:within 30) sigalrm-status))

  (test-case "a Racket timeout does not end a process wedged in atomic mode"
    (check-false (run '(sync/timeout 1 (thread wedge)) #:within 5)))

  (test-case "the deadline is lifted when its body returns"
    (check-equal? (run '(begin (with-process-deadline 1 'done) (sleep 2)) #:within 30) 0))

  (test-case "a custodian holding a running parallel thread shuts down"
    (check-equal? (run '(let ([c (make-custodian)])
                          (busy-worker-under c)
                          (sleep 0.1)
                          (with-process-deadline 10 (shut-down-custodian! c)))
                       #:within 30)
                  0)))

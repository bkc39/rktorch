#lang racket/base

(require (only-in ffi/unsafe _fun _uint get-ffi-obj)
         (only-in racket/port port->string))

(provide arm-process-deadline!
         call-with-process-deadline
         exit-status-of)

;; SIGALRM ends a process wedged in atomic mode, and a process has one alarm.
(define alarm (get-ffi-obj "alarm" #f (_fun _uint -> _uint)))

(define (arm-process-deadline! seconds)
  (define outer (alarm seconds))
  (define armed-at (current-milliseconds))
  (define (outer-left)
    (max 1 (- outer (quotient (- (current-milliseconds) armed-at) 1000))))
  (cond
    [(zero? outer) (lambda () (void (alarm 0)))]
    [else
     (alarm (min seconds outer))
     (lambda () (void (alarm (outer-left))))]))

(define (call-with-process-deadline seconds thunk)
  (define lift! #f)
  (dynamic-wind (lambda () (set! lift! (arm-process-deadline! seconds)))
                thunk
                (lambda () (lift!))))

(define (exit-status-of program #:within seconds)
  (define-values (sp out in err)
    (subprocess #f #f #f (find-system-path 'exec-file) "-e" (format "~s" program)))
  (close-output-port in)
  (define drains (for/list ([p (list out err)]) (thread (lambda () (port->string p)))))
  (define status
    (cond
      [(sync/timeout seconds sp) (subprocess-status sp)]
      [else (subprocess-kill sp #t) #f]))
  (for ([t (in-list drains)]) (sync/timeout 10 t))
  (close-input-port out)
  (close-input-port err)
  status)

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
          ,body)
        (require 'run))
     #:within seconds))

  (test-case "a process wedged in atomic mode ends at its deadline"
    (check-equal? (run '(call-with-process-deadline 1 wedge) #:within 30) sigalrm-status))

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
                  sigalrm-status)))

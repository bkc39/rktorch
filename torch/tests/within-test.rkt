#lang racket/base

(module+ test
  (require (only-in ffi/unsafe/custodian make-custodian-at-root)
           (only-in racket/function const)
           rackunit
           (only-in "private/wedge.rkt" status-and-stderr-of)
           (only-in "private/within.rkt" within))

  (define pools '(#f own))

  (define (start pool step)
    (define (loop) (step) (loop))
    (if pool (thread loop #:pool pool) (thread loop)))

  (define (timed-out thunk)
    (with-handlers ([exn:test:check? (const 'timed-out)])
      (within 0.2 thunk)))

  (define (stepping? count)
    (define before (unbox count))
    (sleep 0.2)
    (not (= before (unbox count))))

  (for ([pool (in-list pools)])
    (test-case (format "a timed-out within stops the workers its body started (pool ~a)" pool)
      (define count (box 0))
      (define orphan (box #f))
      (check-eq? (timed-out (lambda ()
                              (set-box! orphan
                                        (start pool (lambda ()
                                                      (make-vector 100)
                                                      (set-box! count (add1 (unbox count))))))
                              (sync never-evt)))
                 'timed-out)
      (check-true (thread-dead? (unbox orphan)) "the orphan is still running")
      (check-false (stepping? count) "the orphan is still stepping")))

  (test-case "a timed-out body unwinds, after the workers it started are stopped"
    (define orphan (box #f))
    (define seen-at-unwind (box 'not-unwound))
    (check-eq? (timed-out (lambda ()
                            (set-box! orphan (start 'own (lambda () (make-vector 100))))
                            (dynamic-wind void
                                          (lambda () (sync never-evt))
                                          (lambda ()
                                            (set-box! seen-at-unwind
                                                      (thread-dead? (unbox orphan)))))))
               'timed-out)
    (check-equal? (unbox seen-at-unwind) #t))

  (test-case "a thread the body starts at the root outlives the timeout"
    (define count (box 0))
    (define root (box #f))
    (check-eq? (timed-out (lambda ()
                            (set-box! root (make-custodian-at-root))
                            (parameterize ([current-custodian (unbox root)])
                              (start #f (lambda ()
                                          (sleep 0.01)
                                          (set-box! count (add1 (unbox count))))))
                            (sync never-evt)))
               'timed-out)
    (check-true (stepping? count))
    (custodian-shutdown-all (unbox root)))

  (test-case "within returns the body's values and re-raises what it raised"
    (check-equal? (call-with-values (lambda () (within 5 (lambda () (values 1 2)))) list)
                  '(1 2))
    (check-exn #rx"boom" (lambda () (within 5 (lambda () (error 'body "boom"))))))

  (test-case "workers the body leaves running after it returns are not stopped"
    (define count (box 0))
    (define worker
      (within 5 (lambda ()
                  (start #f (lambda ()
                              (sleep 0.01)
                              (set-box! count (add1 (unbox count))))))))
    (check-true (stepping? count))
    (kill-thread worker))

  (define within-module
    (path->string (collection-file-path "within.rkt" "torch" "tests" "private")))

  (test-case "a timeout whose worker sits in a foreign call ends the process, named, at its bound"
    (define-values (status stderr)
      (status-and-stderr-of
       `(begin
          (module run racket/base
            (require (only-in ffi/unsafe _fun _uint get-ffi-obj)
                     (file ,within-module))
            (define c-sleep (get-ffi-obj "sleep" #f (_fun #:blocking? #t _uint -> _uint)))
            (within 2
                    (lambda ()
                      (define inside (make-semaphore 0))
                      (thread (lambda () (semaphore-post inside) (c-sleep 600)) #:pool 'own)
                      (semaphore-wait inside)
                      (sync never-evt))
                    #:unwind-seconds 2))
          (require 'run))
       #:within 30))
    (check-equal? status (+ 128 14) stderr)
    (check-regexp-match
     #rx"process deadline: stopping the threads of a body with no result within 2 s ran past 2 s"
     stderr)))

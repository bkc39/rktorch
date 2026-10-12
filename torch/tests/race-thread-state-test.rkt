#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../foreign.rkt"
                    * copy! matmul ones requires-grad? tensor-dtype
                    with-autocast with-no-grad zeros)
           (only-in "../foreign/error.rkt" check-handle)
           (only-in "../foreign/raw/linalg.rkt" tr-mm/raw)
           (only-in "../foreign/raw/memory.rkt" oom-retry)
           (only-in "private/race-harness.rkt"
                    check-known-failure join labelled make-gate release-gate!
                    spawn wait-for-arrival while-inside with-parked))

  (define (failure-message thunk)
    (with-handlers ([exn:fail? exn-message])
      (thunk)
      #f))

  (define ((failing-mm rows inner))
    (check-handle 'mm (tr-mm/raw (zeros rows inner) (zeros (add1 inner) 2))))

  (define (kind pool) (or pool 'ordinary))

  (define (clobbered-failure pool)
    (define g (make-gate 'failure-reading #:when (labelled 'a)))
    (with-parked [a (spawn (lambda () (failure-message (failing-mm 2 3)))
                           #:pool pool #:label 'a)
                  #:at g]
      (check-eq? (wait-for-arrival g) 'parked)
      (define other
        (join (spawn (lambda () (failure-message (failing-mm 2 5)))
                     #:pool pool #:label 'b)))
      (release-gate! g)
      (define own (join a))
      (check-regexp-match #rx"2x5 and 6x2" other)
      own))

  (test-case "a failure's message survives another thread's failure"
    (for ([pool (in-list '(#f own))])
      (define own (clobbered-failure pool))
      (check-known-failure
       "#194"
       (format "the parked thread reads its own error (~a)" (kind pool))
       (regexp-match? #rx"2x3 and 4x2" own))))

  (define (clobbered-oom-kind pool)
    (define last-kind (box #f))
    (define retried? (box #f))
    (define ((failing error-kind))
      (set-box! last-kind error-kind)
      #f)
    (define (retrying error-kind)
      ((oom-retry #:oom? (lambda () (eq? (unbox last-kind) 'oom))
                  #:collect! (lambda () (set-box! retried? #t)))
       (failing error-kind)))
    (define g (make-gate 'oom-kind-reading #:when (labelled 'a)))
    (with-parked [a (spawn (retrying 'oom) #:pool pool #:label 'a) #:at g]
      (wait-for-arrival g)
      (join (spawn (retrying 'other) #:pool pool #:label 'b))
      (release-gate! g)
      (join a)
      (unbox retried?)))

  (test-case "an OOM's kind survives another thread's failure"
    (for ([pool (in-list '(#f own))])
      (check-known-failure
       "#194"
       (format "the parked thread retries its own OOM (~a)" (kind pool))
       (clobbered-oom-kind pool))))

  (define ((failing-copy width))
    (copy! (zeros 2 2) (zeros width width)))

  (test-case "a failing in-place call on a parallel thread reports its own error"
    (check-regexp-match #rx"tensor b [(]5[)]" (failure-message (failing-copy 5)))
    (define own
      (join (spawn (lambda () (failure-message (failing-copy 3))) #:pool 'own)))
    (check-regexp-match #rx"^copy!: " own)
    (check-known-failure
     "#194"
     "the error read after a plain call on a parallel thread is that call's"
     (regexp-match? #rx"tensor b [(]3[)]" own)))

  (define x (ones 2 2 #:requires-grad? #t))

  (define (grad-tracked?)
    (requires-grad? (* x 2.0)))

  (define (matmul-dtype)
    (tensor-dtype (matmul (ones 4 4) (ones 4 4))))

  (define (in-worker pool thunk)
    (join (spawn thunk #:pool pool)))

  (define (no-grad thunk) (with-no-grad (thunk)))
  (define (autocast thunk) (with-autocast #:device 'cpu (thunk)))

  (struct mode (name enter observe outside inside))

  (define modes
    (list (mode 'grad no-grad grad-tracked? #t #f)
          (mode 'autocast autocast matmul-dtype 'float32 'bfloat16)))

  (struct outcome (description holds? known-failure?))

  (define (mode-leaks pool m)
    (match-define (mode name enter observe outside inside) m)
    (define own? (eq? pool 'own))
    (list
     (outcome (format "a worker's ~a mode stays out of the trainer (~a)"
                      name (kind pool))
              (equal? outside
                      (while-inside enter (lambda (_w) (observe)) #:pool pool))
              (not own?))
     (outcome (format "the trainer's ~a mode stays out of a worker (~a)"
                      name (kind pool))
              (equal? outside (enter (lambda () (in-worker pool observe))))
              #t)
     (outcome (format "a worker's ~a mode reaches its own ops (~a)"
                      name (kind pool))
              (equal? inside (in-worker pool (lambda () (enter observe))))
              own?)))

  (test-case "grad mode and autocast stay with the thread that set them"
    (check-true (grad-tracked?))
    (check-eq? (matmul-dtype) 'float32)
    (for* ([pool (in-list '(#f own))]
           [m (in-list modes)]
           [o (in-list (mode-leaks pool m))])
      (match-define (outcome description holds? known-failure?) o)
      (cond
        [known-failure? (check-known-failure "#194" description holds?)]
        [else (check-true holds? description)]))
    (check-true (grad-tracked?) "a case left grad mode off")
    (check-eq? (matmul-dtype) 'float32 "a case left autocast on")))

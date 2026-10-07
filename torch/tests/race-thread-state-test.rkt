#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    copy! matmul mul ones requires-grad? tensor-dtype
                    with-autocast with-no-grad zeros)
           (only-in "../foreign/error.rkt" check-handle)
           (only-in "../foreign/raw/linalg.rkt" tr-mm/raw)
           (only-in "private/race-harness.rkt"
                    check-known-failure gate-hook join labelled make-gate
                    release-gate! spawn wait-for-arrival with-race-hook))

  (define (failure-message thunk)
    (with-handlers ([exn:fail? exn-message])
      (thunk)
      #f))

  (define ((failing-mm rows inner))
    (check-handle 'mm (tr-mm/raw (zeros rows inner) (zeros (add1 inner) 2))))

  (define (kind pool) (or pool 'ordinary))

  (define (clobbered-failure pool)
    (define g (make-gate 'failure-reading #:when (labelled 'a)))
    (with-race-hook (gate-hook g)
      (define a
        (spawn (lambda () (failure-message (failing-mm 2 3))) #:pool pool #:label 'a))
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
    (requires-grad? (mul x 2.0)))

  (define (matmul-dtype)
    (tensor-dtype (matmul (ones 4 4) (ones 4 4))))

  (define (in-worker pool thunk)
    (join (spawn thunk #:pool pool)))

  (define (while-worker-inside pool enter observe)
    (define inside (make-semaphore 0))
    (define leave (make-semaphore 0))
    (define w
      (spawn (lambda ()
               (enter (lambda ()
                        (semaphore-post inside)
                        (semaphore-wait leave))))
             #:pool pool))
    (check-not-false (sync/timeout 30 inside))
    (define seen (observe))
    (semaphore-post leave)
    (join w)
    seen)

  (define (no-grad thunk) (with-no-grad (thunk)))
  (define (autocast thunk) (with-autocast #:device 'cpu (thunk)))

  (define mode-cases
    (list (list 'grad no-grad grad-tracked? #t #f)
          (list 'autocast autocast matmul-dtype 'float32 'bfloat16)))

  (define (mode-leaks pool name enter observe outside inside)
    (list
     (list (format "a worker's ~a mode stays out of the trainer (~a)"
                   name (kind pool))
           (equal? outside (while-worker-inside pool enter observe)))
     (list (format "the trainer's ~a mode stays out of a worker (~a)"
                   name (kind pool))
           (equal? outside (enter (lambda () (in-worker pool observe)))))
     (list (format "a worker's ~a mode reaches its own ops (~a)" name (kind pool))
           (equal? inside (in-worker pool (lambda () (enter observe)))))))

  (define holds-today
    '("a worker's grad mode stays out of the trainer (own)"
      "a worker's grad mode reaches its own ops (ordinary)"
      "a worker's autocast mode stays out of the trainer (own)"
      "a worker's autocast mode reaches its own ops (ordinary)"))

  (test-case "grad mode and autocast stay with the thread that set them"
    (check-true (grad-tracked?))
    (check-eq? (matmul-dtype) 'float32)
    (for* ([pool (in-list '(#f own))]
           [mode (in-list mode-cases)]
           [outcome (in-list (apply mode-leaks pool mode))])
      (define description (car outcome))
      (define holds? (cadr outcome))
      (if (member description holds-today)
          (check-true holds? description)
          (check-known-failure "#194" description holds?)))
    (check-true (grad-tracked?) "a case left grad mode off")
    (check-eq? (matmul-dtype) 'float32 "a case left autocast on")))

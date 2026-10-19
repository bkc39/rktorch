#lang racket/base

(module+ test
  (require (only-in ffi/unsafe _fun _int _uint get-ffi-obj)
           rackunit
           "../data/results-queue.rkt")

  (define (within seconds thunk)
    (define outcome #f)
    (define t
      (thread (lambda ()
                (set! outcome
                      (with-handlers ([(lambda (_v) #t) (lambda (v) (lambda () (raise v)))])
                        (define v (thunk))
                        (lambda () v))))))
    (cond
      [(sync/timeout seconds t) (outcome)]
      [else
       (kill-thread t)
       (fail (format "no result within ~a s" seconds))]))

  (define (take q r [abandon never-evt])
    (within 10 (lambda () (results-queue-take! q r abandon #f))))

  (test-case "a put result is taken"
    (define q (results-queue))
    (results-queue-put! q 3 'three)
    (check-equal? (take q 3) 'three))

  (test-case "take waits for a put from a parallel thread"
    (define q (results-queue))
    (define got #f)
    (define taker (thread (lambda () (set! got (results-queue-take! q 0 never-evt #f)))))
    (check-false (sync/timeout 0.1 taker))
    (define worker (thread (lambda () (results-queue-put! q 0 'zero)) #:pool 'own))
    (check-not-false (sync/timeout 10 worker))
    (check-not-false (sync/timeout 10 taker))
    (check-equal? got 'zero))

  (test-case "out-of-order puts are taken in index order"
    (define q (results-queue))
    (for ([r (in-list '(2 0 1))])
      (results-queue-put! q r (* 10 r)))
    (check-equal? (for/list ([r (in-range 3)]) (take q r)) '(0 10 20)))

  (test-case "take answers #f once the abandon event is ready and the result is absent"
    (define q (results-queue))
    (results-queue-put! q 1 'one)
    (check-false (take q 0 always-evt)))

  (test-case "a result put before the abandon event fires is still taken"
    (define q (results-queue))
    (results-queue-put! q 0 'zero)
    (check-equal? (take q 0 always-evt) 'zero)
    (for ([r (in-range 1 41)])
      (define gone (make-semaphore 0))
      (define got #f)
      (define taker
        (thread (lambda ()
                  (set! got (results-queue-take! q r (semaphore-peek-evt gone) #f)))))
      (check-false (sync/timeout 0.01 taker))
      (results-queue-put! q r r)
      (semaphore-post gone)
      (check-not-false (sync/timeout 10 taker))
      (check-equal? got r)))

  (test-case "take removes the entry"
    (define q (results-queue))
    (results-queue-put! q 0 'zero)
    (check-equal? (take q 0) 'zero)
    (check-false (take q 0 always-evt)))

  (test-case "a breakable take can be broken while it waits"
    (define q (results-queue))
    (define outcome #f)
    (define taker
      (thread (lambda ()
                (with-handlers ([exn:break? (lambda (e) (set! outcome e))])
                  (parameterize-break #f
                    (results-queue-take! q 0 never-evt #t))))))
    (check-false (sync/timeout 0.1 taker))
    (break-thread taker)
    (check-not-false (sync/timeout 10 taker))
    (check-pred exn:break? outcome))

  (test-case "puts complete while the main thread is inside a native call"
    (define usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))
    (define q (results-queue))
    (define started (make-semaphore 0))
    (define deposited (box 0))
    (define worker
      (thread (lambda ()
                (semaphore-post started)
                (for ([r (in-range 100)])
                  (results-queue-put! q r r)
                  (set-box! deposited (add1 r))))
              #:pool 'own))
    (check-not-false (sync/timeout 10 started))
    (define seen (begin (usleep 500000) (unbox deposited)))
    (check-equal? seen 100)
    (check-not-false (sync/timeout 10 worker))
    (check-equal? (for/list ([r (in-range 100)]) (take q r)) (for/list ([r (in-range 100)]) r))))

#lang racket/base

(module+ test
  (require (only-in ffi/unsafe _fun _int _uint get-ffi-obj)
           (only-in racket/list range)
           rackunit
           "../data/results-queue.rkt"
           (only-in "private/within.rkt" within))

  (define (take q index [abandon never-evt])
    (within 10 (lambda () (results-queue-take! q index #:abandon abandon #:breakable? #f))))

  (test-case "a put result is taken"
    (define q (results-queue))
    (results-queue-put! q 3 'three)
    (check-equal? (take q 3) 'three))

  (test-case "take waits for a put from a parallel thread"
    (define q (results-queue))
    (define got #f)
    (define taker
      (thread (lambda ()
                (set! got (results-queue-take! q 0 #:abandon never-evt #:breakable? #f)))))
    (check-false (sync/timeout 0.1 taker))
    (define worker (thread (lambda () (results-queue-put! q 0 'zero)) #:pool 'own))
    (check-not-false (sync/timeout 10 worker))
    (check-not-false (sync/timeout 10 taker))
    (check-equal? got 'zero))

  (test-case "out-of-order puts are taken in index order"
    (define q (results-queue))
    (for ([index (in-list '(2 0 1))])
      (results-queue-put! q index (* 10 index)))
    (check-equal? (for/list ([index (in-range 3)]) (take q index)) '(0 10 20)))

  (test-case "take answers #f once the abandon event is ready and the result is absent"
    (define q (results-queue))
    (results-queue-put! q 1 'one)
    (check-false (take q 0 always-evt)))

  (test-case "a result put before the abandon event fires is still taken"
    (define q (results-queue))
    (results-queue-put! q 0 'zero)
    (check-equal? (take q 0 always-evt) 'zero)
    (define dropped
      (for/list ([index (in-range 1 201)]
                 #:unless (take q index (guard-evt (lambda ()
                                                     (results-queue-put! q index index)
                                                     always-evt))))
        index))
    (check-equal? dropped '()))

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
                    (results-queue-take! q 0 #:abandon never-evt #:breakable? #t))))))
    (check-false (sync/timeout 0.1 taker))
    (break-thread taker)
    (check-not-false (sync/timeout 10 taker))
    (check-pred exn:break? outcome))

  (test-case "a non-breakable take keeps waiting through a break"
    (define q (results-queue))
    (define got #f)
    (define taker
      (thread (lambda ()
                (with-handlers ([exn:break? void])
                  (parameterize-break #f
                    (set! got (results-queue-take! q 0 #:abandon never-evt #:breakable? #f)))))))
    (check-false (sync/timeout 0.1 taker))
    (break-thread taker)
    (check-false (sync/timeout 0.1 taker))
    (results-queue-put! q 0 'zero)
    (check-not-false (sync/timeout 10 taker))
    (check-equal? got 'zero))

  (test-case "puts complete while the main thread is inside a native call"
    (define usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))
    (define q (results-queue))
    (define started (make-semaphore 0))
    (define deposited (box 0))
    (collect-garbage 'minor)
    (define worker
      (thread (lambda ()
                (semaphore-post started)
                (for ([index (in-range 100)])
                  (results-queue-put! q index index)
                  (set-box! deposited (add1 index))))
              #:pool 'own))
    (check-not-false (sync/timeout 10 started))
    (define seen (begin (usleep 500000) (unbox deposited)))
    (check-equal? seen 100)
    (check-not-false (sync/timeout 10 worker))
    (check-equal? (for/list ([index (in-range 100)]) (take q index)) (range 100))))

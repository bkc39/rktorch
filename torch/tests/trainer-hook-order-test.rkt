#lang racket/base

(module+ test
  (require (only-in racket/list take)
           (only-in racket/match match*)
           rackunit
           "../train.rkt"
           (only-in "private/lightning-names.rkt" expected-events)
           (only-in "private/trainer-kit.rkt"
                    recorder Recorded scenario scenario-events scenario-exn toy-loader))

  (define (same-event? a e)
    (match* (a e)
      [((list source hook epoch step batch lr-a) (list source hook epoch step batch lr-e))
       (or (and (not lr-a) (not lr-e))
           (and lr-a lr-e (< (abs (- lr-a lr-e)) 1e-12)))]
      [(_ _) #f]))

  (define (check-events name sc)
    (define actual (scenario-events sc))
    (define expected (expected-events name))
    (define mismatch
      (for/first ([a (in-list actual)] [e (in-list expected)] [i (in-naturals)]
                  #:unless (same-event? a e))
        (format "~a: event ~a is ~s, Lightning's is ~s" name i a e)))
    (check-false mismatch (or mismatch ""))
    (check-equal? (length actual) (length expected)
                  (format "~a: event count; the shorter list ends ~s" name
                          (take (if (< (length actual) (length expected)) actual expected)
                                (min 3 (length actual) (length expected))))))

  (define (fit-scenario sc
                        #:train [n 6]
                        #:accumulate [accumulate 1]
                        #:val-check-interval [interval #f])
    (fit (Recorded sc) (toy-loader n)
         #:val (toy-loader 4)
         #:max-epochs 2
         #:num-sanity-val-steps 1
         #:accumulate-grad-batches accumulate
         #:val-check-interval interval
         #:logger #f
         #:callbacks (list (recorder sc))))

  (test-case "LO-1: fit with a validation loader"
    (define sc (scenario))
    (fit-scenario sc)
    (check-events 'fit-val sc))

  (test-case "LO-2: two-batch accumulation with a per-step scheduler"
    (define sc (scenario #:step-scheduler? #t))
    (fit-scenario sc #:train 10 #:accumulate 2)
    (check-events 'accumulate sc))

  (test-case "LO-3: skip-epoch! at epoch 0, batch 1"
    (define sc (scenario #:skip-epoch '(0 1)))
    (fit-scenario sc)
    (check-events 'skip-epoch sc))

  (test-case "LO-4: an exn:fail in the training step at epoch 0, batch 1"
    (define sc (scenario #:fail-at '(0 1)))
    (check-exn #rx"planned failure" (lambda () (fit-scenario sc)))
    (check-events 'exception sc))

  (test-case "LO-5: request-stop! at batch 1"
    (define sc (scenario #:stop-at '(0 1)))
    (fit-scenario sc)
    (check-events 'request-stop sc))

  (test-case "LO-6: an integer #:val-check-interval"
    (define sc (scenario))
    (fit-scenario sc #:train 10 #:val-check-interval 2)
    (check-events 'val-check-interval sc))

  (test-case "LO-7: test"
    (define sc (scenario))
    (test (Recorded sc) (toy-loader 4) #:logger #f #:callbacks (list (recorder sc)))
    (check-events 'test sc))

  (test-case "LO-8: predict"
    (define sc (scenario))
    (predict (Recorded sc) (toy-loader 4) #:callbacks (list (recorder sc)))
    (check-events 'predict sc))

  (test-case "LO-9: an exn:break runs LO-4's hooks, and a second break is absorbed"
    (define sc (scenario #:fail-at '(0 1) #:break-twice? #t))
    (define caught #f)
    (define late
      (with-handlers ([exn:break? values])
        (set! caught (with-handlers ([exn:break? values]) (fit-scenario sc)))
        (sleep 0)
        #f))
    (check-pred exn:break? caught)
    (check-eq? caught (scenario-exn sc) "fit re-raises the break on-exception received")
    (check-false late "the break issued inside on-exception reached the caller")
    (check-events 'exception sc)))

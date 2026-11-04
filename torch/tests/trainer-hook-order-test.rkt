#lang racket/base

(module+ test
  (require (only-in racket/list take)
           (only-in racket/match match*)
           rackunit
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/lightning-names.rkt" expected-events expected-logged-steps)
           (only-in "private/trainer-kit.rkt"
                    collecting-logger collecting-rows recorder Recorded scenario
                    scenario-events scenario-exn toy-loader)
           (only-in "private/within.rkt" bounded-test))

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
                        #:val-check-interval [interval #f]
                        #:max-epochs [max-epochs 2]
                        #:max-steps [max-steps #f])
    (fit (Recorded sc) (toy-loader n)
         #:val (toy-loader 4)
         #:max-epochs max-epochs
         #:max-steps max-steps
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

  (test-case "LO-11: a #f loss skips its backward pass; the window still steps"
    (define sc (scenario #:skip-loss '(0 1)))
    (fit-scenario sc)
    (check-events 'skip-loss sc))

  (test-case "LO-11: skip-step! drops the step, not the backward passes or the step count"
    (define sc (scenario #:skip-step '(0 1)))
    (fit-scenario sc)
    (check-events 'skip-step sc))

  (test-case "LO-11: a #f loss at a window's start, under a per-step scheduler"
    (define sc (scenario #:skip-loss '(0 0) #:step-scheduler? #t))
    (fit-scenario sc #:train 10 #:accumulate 2)
    (check-events 'skip-loss-accumulate sc))

  (test-case "LO-11: a #f loss at a window's end, under a per-step scheduler"
    (define sc (scenario #:skip-loss '(0 1) #:step-scheduler? #t))
    (fit-scenario sc #:train 10 #:accumulate 2)
    (check-events 'skip-loss-end-accumulate sc))

  (test-case "LO-11: a #f loss in the last, one-batch window, under a per-step scheduler"
    (define sc (scenario #:skip-loss '(0 4) #:step-scheduler? #t))
    (fit-scenario sc #:train 10 #:accumulate 2)
    (check-events 'skip-loss-last-accumulate sc))

  (test-case "LO-11: a two-batch window whose every loss is #f, under a per-step scheduler"
    (define sc (scenario #:skip-loss '((0 0) (0 1)) #:step-scheduler? #t))
    (fit-scenario sc #:train 10 #:accumulate 2)
    (check-events 'skip-window-accumulate sc))

  (test-case "LO-11: a per-step scheduler steps on a window skip-step! dropped"
    (define sc (scenario #:skip-step '(0 1) #:step-scheduler? #t))
    (fit-scenario sc #:train 10 #:accumulate 2)
    (check-events 'skip-step-accumulate sc))

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
    (check-events 'exception sc))

  (define-trainer Skipping (net skip)
    (define-init (s) (set! net (Linear 2 1)) (set! skip s))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (log! 'tl 1.0 #:on-step? #t #:on-epoch? #t)
      (define loss (mse-loss (net x) y))
      (define here? (equal? (list (run-epoch run) (run-batch-index run)) '(0 1)))
      (cond
        [(and here? (eq? skip 'loss)) #f]
        [(and here? (eq? skip 'step)) (skip-step! run) loss]
        [else loss]))
    (define-validation-step (x y)
      (log! 'vl 2.0)
      (mse-loss (net x) y)))

  (define (check-logged-steps name skip #:train [n 6] #:accumulate [accumulate 1]
                              #:log-every-n-steps [log-every 1])
    (define l (collecting-logger))
    (fit (Skipping skip) (toy-loader n)
         #:val (toy-loader 4)
         #:max-epochs 2
         #:num-sanity-val-steps 1
         #:accumulate-grad-batches accumulate
         #:log-every-n-steps log-every
         #:logger l)
    (check-equal? (collecting-rows l) (expected-logged-steps name) (format "~a" name)))

  (test-case "LO-10: a skipped window advances the logged step, by a #f loss or skip-step!"
    (check-logged-steps 'skip-loss 'loss)
    (check-logged-steps 'skip-step 'step)
    (check-logged-steps 'skip-loss-accumulate 'loss #:train 10 #:accumulate 2)
    (check-logged-steps 'skip-loss-log-every-2 'loss #:log-every-n-steps 2))

  (bounded-test "LO-11: #:max-steps alone ends a fit whose every loss is #f"
    (define sc (scenario #:skip-loss 'every))
    (fit-scenario sc #:max-epochs #f #:max-steps 4)
    (check-events 'skip-every-max-steps sc)))

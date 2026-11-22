#lang racket/base

(require (only-in racket/list take)
         (only-in racket/match match*)
         rackunit
         "../../train.rkt"
         (only-in "lightning-names.rkt" expected-events)
         (only-in "trainer-kit.rkt" recorder Recorded scenario-events toy-loader))

(provide check-events epoch-saver fit-scenario step-saver)

(define (save-here! run dir weights-only?)
  (save-checkpoint! run
                    (build-path dir (format "epoch=~a-step=~a.ckpt" (run-epoch run) (run-global-step run)))
                    #:weights-only? weights-only?))

(define-callback epoch-saver (dir weights-only?)
  #:kind checkpoint
  (define-init (d #:weights-only? [w? #f]) (set! dir d) (set! weights-only? w?))
  (define-hook (on-train-epoch-end) (save-here! run dir weights-only?)))

(define-callback step-saver (dir every)
  #:kind checkpoint
  (define-hook (on-train-batch-end _loss _extra . _batch)
    (when (zero? (modulo (run-global-step run) every))
      (save-here! run dir #f))))

(define (same-event? a e)
  (match* (a e)
    [((list source hook epoch step batch lr-a) (list source hook epoch step batch lr-e))
     (or (and (not lr-a) (not lr-e))
         (and lr-a lr-e (< (abs (- lr-a lr-e)) 1e-12)))]
    [(_ _) #f]))

(define (check-events name sc #:expected [expected (expected-events name)])
  (define actual (scenario-events sc))
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
                      #:trainer [make Recorded]
                      #:callbacks [callbacks '()]
                      #:precision [precision '32-true]
                      #:train [n 6]
                      #:accumulate [accumulate 1]
                      #:val-check-interval [interval #f]
                      #:max-epochs [max-epochs 2]
                      #:max-steps [max-steps #f])
  (fit (make sc) (toy-loader n)
       #:val (toy-loader 4)
       #:max-epochs max-epochs
       #:max-steps max-steps
       #:num-sanity-val-steps 1
       #:accumulate-grad-batches accumulate
       #:val-check-interval interval
       #:precision precision
       #:logger #f
       #:callbacks (cons (recorder sc) callbacks)))

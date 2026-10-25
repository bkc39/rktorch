#lang racket/base

(require (only-in racket/contract/base
                  ->i >/c listof or/c unsupplied-arg?)
         (only-in racket/match match)
         (only-in racket/math exact-floor)
         (only-in "../data/loader.rkt" dataloader-length dataloader?)
         (only-in "../foreign.rkt" device?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "logging.rkt" logger-spec/c stdout-logger)
         (only-in "private/types.rkt"
                  callback? not-implemented trainer-automatic?
                  trainer-defines? trainer?)
         (only-in "run.rkt"
                  clip-algorithm/c fast-dev-run/c limit/c precision/c run?
                  val-check-interval/c))

(define (given v [default #f])
  (if (unsupplied-arg? v) default v))

(define (limited-batches loader limit)
  (define n (dataloader-length loader))
  (match (given limit 1.0)
    [(? exact-integer? k) (min k n)]
    [share (exact-floor (* n share))]))

(define (selects-a-batch? loader limit)
  (positive? (limited-batches loader limit)))

(define/contract-out (fit trainer train
                          #:val [val #f]
                          #:max-epochs [max-epochs #f]
                          #:max-steps [max-steps #f]
                          #:device [device #f]
                          #:logger [logger (stdout-logger)]
                          #:callbacks [callbacks '()]
                          #:accumulate-grad-batches [accumulate 1]
                          #:gradient-clip-val [clip-val #f]
                          #:gradient-clip-algorithm [clip-algorithm 'norm]
                          #:precision [precision '32-true]
                          #:check-val-every-n-epoch [val-every 1]
                          #:val-check-interval [val-interval #f]
                          #:num-sanity-val-steps [sanity 2]
                          #:limit-train-batches [limit-train 1.0]
                          #:limit-val-batches [limit-val 1.0]
                          #:log-every-n-steps [log-every 50]
                          #:fast-dev-run [fast-dev-run #f]
                          #:ckpt-path [ckpt-path #f])
  (->i ([trainer trainer?] [train dataloader?])
       (#:val [val (or/c #f dataloader?)]
        #:max-epochs [max-epochs (or/c #f exact-positive-integer?)]
        #:max-steps [max-steps (or/c #f exact-positive-integer?)]
        #:device [device (or/c #f device?)]
        #:logger [logger logger-spec/c]
        #:callbacks [callbacks (listof callback?)]
        #:accumulate-grad-batches [accumulate exact-positive-integer?]
        #:gradient-clip-val [clip-val (or/c #f (>/c 0))]
        #:gradient-clip-algorithm [clip-algorithm clip-algorithm/c]
        #:precision [precision precision/c]
        #:check-val-every-n-epoch [val-every exact-positive-integer?]
        #:val-check-interval [val-interval val-check-interval/c]
        #:num-sanity-val-steps [sanity exact-nonnegative-integer?]
        #:limit-train-batches [limit-train limit/c]
        #:limit-val-batches [limit-val limit/c]
        #:log-every-n-steps [log-every exact-positive-integer?]
        #:fast-dev-run [fast-dev-run fast-dev-run/c]
        #:ckpt-path [ckpt-path (or/c #f path-string?)])
       #:pre/name (max-epochs max-steps fast-dev-run)
       "fit needs #:max-epochs or #:max-steps"
       (or (given max-epochs) (given max-steps) (given fast-dev-run))
       #:pre/name (trainer) "the trainer has no define-training-step"
       (trainer-defines? trainer 'define-training-step)
       #:pre/name (trainer) "automatic optimization needs define-optimizers"
       (or (not (trainer-automatic? trainer))
           (trainer-defines? trainer 'define-optimizers))
       #:pre/name (trainer val) "a #:val loader needs a define-validation-step"
       (or (not (given val)) (trainer-defines? trainer 'define-validation-step))
       #:pre/name (train limit-train)
       "#:limit-train-batches selects no batch from this loader"
       (selects-a-batch? train limit-train)
       #:pre/name (val limit-val)
       "#:limit-val-batches selects no batch from this loader"
       (or (not (given val)) (selects-a-batch? val limit-val))
       #:pre/name (train limit-train val-interval)
       "an integer #:val-check-interval exceeds the training batches #:limit-train-batches selects"
       (match (given val-interval)
         [(? exact-integer? k) (<= k (limited-batches train limit-train))]
         [_ #t])
       #:pre/name (trainer accumulate clip-val)
       "manual optimization takes no #:accumulate-grad-batches above 1 and no #:gradient-clip-val"
       (or (trainer-automatic? trainer)
           (and (= 1 (given accumulate 1)) (not (given clip-val))))
       [result run?])
  (not-implemented 'fit 'T1))

(define/contract-out (test trainer loader
                           #:device [device #f]
                           #:logger [logger (stdout-logger)]
                           #:callbacks [callbacks '()]
                           #:limit-test-batches [limit 1.0]
                           #:ckpt-path [ckpt-path #f])
  (->i ([trainer trainer?] [loader dataloader?])
       (#:device [device (or/c #f device?)]
        #:logger [logger logger-spec/c]
        #:callbacks [callbacks (listof callback?)]
        #:limit-test-batches [limit limit/c]
        #:ckpt-path [ckpt-path (or/c #f path-string?)])
       #:pre/name (trainer) "the trainer has no define-test-step"
       (trainer-defines? trainer 'define-test-step)
       #:pre/name (loader limit)
       "#:limit-test-batches selects no batch from this loader"
       (selects-a-batch? loader limit)
       [result run?])
  (not-implemented 'test 'T1))

(define/contract-out (predict trainer loader
                              #:device [device #f]
                              #:callbacks [callbacks '()]
                              #:limit-predict-batches [limit 1.0]
                              #:ckpt-path [ckpt-path #f])
  (->i ([trainer trainer?] [loader dataloader?])
       (#:device [device (or/c #f device?)]
        #:callbacks [callbacks (listof callback?)]
        #:limit-predict-batches [limit limit/c]
        #:ckpt-path [ckpt-path (or/c #f path-string?)])
       #:pre/name (trainer) "the trainer has no define-predict-step"
       (trainer-defines? trainer 'define-predict-step)
       [result list?])
  (not-implemented 'predict 'T1))

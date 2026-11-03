#lang racket/base

(require (only-in "book.rkt" make-book))

(provide (struct-out run-struct)
         fresh-run
         live?
         run-loop)

(struct run (trainer
             entry
             device
             book
             callbacks
             loggers
             optimizers
             schedulers
             epoch
             global-step
             batch-size
             eval-batches
             [current-stage #:auto]
             [batch-index #:auto]
             [hook #:auto]
             [stepped? #:auto]
             [step-pending? #:auto]
             [skip-window? #:auto]
             [skip-epoch? #:auto]
             [stop? #:auto]
             [set-up? #:auto]
             [torn-down? #:auto]
             [finished? #:auto])
  #:auto-value #f
  #:mutable
  #:name run-struct
  #:constructor-name make-run)

(define (fresh-run trainer entry device
                   #:callbacks [callbacks '()]
                   #:loggers [loggers '()]
                   #:optimizers [optimizers '()]
                   #:schedulers [schedulers '()]
                   #:epoch [epoch 0]
                   #:global-step [global-step 0]
                   #:batch-size [batch-size 1])
  (make-run trainer entry device (make-book)
            callbacks loggers optimizers schedulers epoch global-step batch-size 0))

(define (live? r) (and (run? r) (not (run-finished? r))))

(define (run-loop r)
  (if (eq? (run-current-stage r) 'train) 'train 'eval))

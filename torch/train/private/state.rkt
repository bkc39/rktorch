#lang racket/base

(require (only-in "book.rkt" make-book))

(provide (struct-out run-struct)
         fresh-run
         live?)

(struct run (trainer
             entry
             device
             book
             [callbacks #:mutable]
             [loggers #:mutable]
             [stage #:mutable]
             [epoch #:mutable]
             [global-step #:mutable]
             [batch-index #:mutable]
             [batch-size #:mutable]
             [optimizers #:mutable]
             [schedulers #:mutable]
             [hook #:mutable]
             [stepped? #:mutable]
             [step-pending? #:mutable]
             [skip-window? #:mutable]
             [skip-epoch? #:mutable]
             [stop? #:mutable]
             [set-up? #:mutable]
             [torn-down? #:mutable]
             [active? #:mutable])
  #:name run-struct
  #:constructor-name make-run)

(define (fresh-run trainer entry device)
  (make-run trainer entry device (make-book)
            '() '() #f 0 0 #f 1 '() '() #f #f #f #f #f #f #f #f #t))

(define (live? r) (and (run? r) (run-active? r)))

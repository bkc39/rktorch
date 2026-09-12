#lang racket/base

(module+ test
  (require rackunit
           "../data/loader.rkt"
           "../main.rkt")

  (test-case "length is len: core containers keep racket's answer"
    (check-equal? (length '(1 2 3)) 3)
    (check-equal? (length '()) 0)
    (check-equal? (length (vector 1 2)) 2)
    (check-equal? (length "four") 4)
    (check-equal? (length (hash 'a 1)) 1)
    (check-true (sized? '()))
    (check-false (sized? 5))
    (check-exn #rx"^length: contract violation" (lambda () (length 5))))

  (test-case "a tensor's length is its first dimension"
    (check-equal? (length (zeros 4 2)) 4)
    (check-equal? (length (zeros 0 2)) 0)
    (check-true (sized? (zeros 1)))
    (check-exn #rx"rank at least one" (lambda () (length (tensor 1.0)))))

  (test-case "datasets and loaders answer length and loaders are sequences"
    (define-dataset squares (n)
      #:init (n)
      #:length n
      #:ref (i) (values (full (* i i) 2) (tensor i)))
    (check-equal? (length (squares 5)) 5)
    (check-equal? (length (tensor-dataset (zeros 6 2))) 6)
    (define loader (dataloader (squares 5) #:batch-size 2))
    (check-equal? (length loader) 3)
    (check-equal? (length (dataloader (squares 5) #:batch-size 2 #:drop-last? #t)) 2)
    (check-equal? (for/list ([(_xb yb) loader]) (tensor->list yb))
                  '((0 1) (2 3) (4))
                  "for over a loader is one epoch")
    (check-equal? (for/list ([(_xb yb) loader]) (tensor->list yb))
                  (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb))
                  "the same traversal as in-dataloader")))

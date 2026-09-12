#lang racket/base

(module+ test
  (require rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           "../data/loader.rkt"
           "../main.rkt")

  ;; class Squares(Dataset):
  ;;     def __init__(self, n): self.n = n
  ;;     def __len__(self): return self.n
  ;;     def __getitem__(self, i): return torch.full((2,), i * i), torch.tensor(i)
  (define-dataset squares (n)
    #:init (n)
    #:length n
    #:ref (i) (values (full (* i i) 2) (tensor i)))

  (test-case "define-dataset: constructor, length, items, printing"
    (define ds (squares 5))
    (check-true (dataset? ds))
    (check-true (squares? ds))
    (check-false (squares? 5))
    (check-equal? (dataset-length ds) 5)
    (define-values (x y) (dataset-ref ds 3))
    (check-equal? (tensor->list x) '(9.0 9.0))
    (check-equal? (tensor->list y) '(3))
    (check-equal? (format "~a" ds) "#<squares>")
    (check-false (dataset-device ds) "no #:device answers #f")
    (define-values (xb yb) (dataset-batch ds '(1 3) default-collate))
    (check-equal? (tensor->list xb) '(1.0 1.0 9.0 9.0) "the default batch collates items")
    (check-equal? (tensor->list yb) '(1 3))
    (check-exn exn:fail:contract? (lambda () (dataset-ref ds -1))))

  (test-case "fields are the constructor formals when there is no #:init"
    (define-dataset pairs (xs ys)
      #:length (length xs)
      #:ref (i) (values (tensor (list-ref xs i)) (tensor (list-ref ys i))))
    (define ds (pairs '(1 2 3) '(4 5 6)))
    (check-equal? (dataset-length ds) 3)
    (define-values (x y) (dataset-ref ds 2))
    (check-equal? (tensor->list x) '(3))
    (check-equal? (tensor->list y) '(6)))

  (test-case "#:init formals take defaults, keywords and a rest argument"
    (define-dataset ramp (n scale offset)
      #:init (n #:scale [scale 1.0] . offset)
      (set! offset (if (null? offset) 0.0 (car offset)))
      #:length n
      #:ref (i) (tensor (+ offset (* scale i))))
    (check-equal? (tensor->list (dataset-ref (ramp 4) 3)) '(3.0))
    (check-equal? (tensor->list (dataset-ref (ramp 4 #:scale 2.0) 3)) '(6.0))
    (check-equal? (tensor->list (dataset-ref (ramp 4 10.0 #:scale 2.0) 3)) '(16.0)))

  (test-case "#:batch and #:device override the fallbacks"
    (define-dataset doubled (n)
      #:init (n)
      #:length n
      #:ref (i) (tensor i)
      #:device (cpu-device)
      #:batch (indices collate)
      (collate (for/list ([i (in-list (indices->list indices))])
                 (list (tensor (* 2 i))))))
    (define ds (doubled 6))
    (check-equal? (dataset-device ds) (cpu-device))
    (define-values (xb) (dataset-batch ds '(1 2 5) default-collate))
    (check-equal? (tensor->list xb) '(2 4 10))
    (define-values (xt) (dataset-batch ds (tensor '(4 0) #:dtype 'int64) default-collate))
    (check-equal? (tensor->list xt) '(8 0) "an index tensor reaches #:batch too")
    (check-equal? (for/list ([xb (in-dataloader (dataloader ds #:batch-size 4))])
                    (tensor->list xb))
                  '((0 2 4 6) (8 10))
                  "a loader goes through #:batch"))

  (test-case "a dataset feeds a loader like any other"
    (define loader (dataloader (squares 5) #:batch-size 2 #:shuffle? #t
                               #:generator (make-generator 0)))
    (check-equal? (dataloader-length loader) 3)
    (define seen
      (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb)))
    (check-equal? (sort (apply append seen) <) '(0 1 2 3 4)))

  (test-case "a struct implementing gen:dataset by hand still works"
    (struct Cubes (n)
      #:methods gen:dataset
      [(define (dataset-length self) (Cubes-n self))
       (define (dataset-ref _self i) (tensor (* i i i)))])
    (check-true (dataset? (Cubes 3)))
    (define-values (xb) (dataset-batch (Cubes 3) '(2) default-collate))
    (check-equal? (tensor->list xb) '(8)))

  (test-case "clauses are checked at expansion"
    (check-exn #rx"#:length clause"
               (lambda ()
                 (convert-compile-time-error
                  (define-dataset no-length (n) #:init (n) #:ref (i) i)))) ;; noqa
    (check-exn #rx"#:ref clause"
               (lambda ()
                 (convert-compile-time-error
                  (define-dataset no-ref (n) #:init (n) #:length n)))) ;; noqa
    (check-exn #rx"needs #:contract"
               (lambda ()
                 (convert-compile-time-error
                  (define-dataset no-ctc (n) #:predicate p? #:init (n) ;; noqa
                    #:length n #:ref (i) i))))
    (check-exn #rx"bare identifier"
               (lambda ()
                 (convert-compile-time-error
                  (define-dataset bad-field ([n 1]) #:init (n) ;; noqa
                    #:length n #:ref (i) i))))))

;; #:contract is a provide, so it lives at module level; the derived
;; predicate is the lowercase name, or the #:predicate one
(module exported racket/base
  (require (only-in racket/contract/base ->)
           "../data/loader.rkt"
           (only-in "../main.rkt" tensor))
  (define-dataset CountUp (n)
    #:contract (-> exact-positive-integer? count-up?)
    #:init (n)
    #:length n
    #:ref (i) (tensor i))
  (define-dataset Named (n)
    #:contract (-> exact-positive-integer? named-dataset?)
    #:predicate named-dataset?
    #:init (n)
    #:length n
    #:ref (i) (tensor i)))

(module+ test
  (require (submod ".." exported))
  (test-case "#:contract exports the constructor and the derived predicate"
    (check-true (count-up? (CountUp 2)))
    (check-true (named-dataset? (Named 2)))
    (check-false (named-dataset? (CountUp 2)))
    (check-exn #rx"CountUp: contract violation" (lambda () (CountUp 0)))
    (check-equal? (dataset-length (CountUp 3)) 3)))

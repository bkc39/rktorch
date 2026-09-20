#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           (only-in "../nn.rkt" cross-entropy nll-loss))

  (test-case "topk defaults to the largest along the last dimension"
    (define-values (top indices) (topk (tensor '((1.0 5.0 3.0) (4.0 2.0 6.0))) 2))
    (check-equal? (tensor-shape top) '(2 2))
    (check-equal? (tensor->list top) '(5.0 3.0 6.0 4.0))
    (check-equal? (tensor->list indices) '(1 2 2 0)))

  (test-case "topk takes the smallest along a named dimension"
    (define-values (top indices)
      (topk (tensor '((1.0 5.0) (4.0 2.0))) 1 #:dim 0 #:largest? #f))
    (check-equal? (tensor->list top) '(1.0 2.0))
    (check-equal? (tensor->list indices) '(0 1)))

  (test-case "a k beyond the dimension is a contract violation"
    (check-exn #rx"k is at most the length of dim"
               (lambda () (topk (tensor '(1.0 2.0 3.0)) 4)))
    (check-exn #rx"k is at most the length of dim"
               (lambda () (topk (tensor '((1.0 2.0 3.0))) 2 #:dim 0))))

  (test-case "sort orders a tensor and answers the permutation"
    (define-values (sorted indices) (sort (tensor '(3.0 1.0 2.0))))
    (check-equal? (tensor->list sorted) '(1.0 2.0 3.0))
    (check-equal? (tensor->list indices) '(1 2 0))
    (define-values (down _) (sort (tensor '(3.0 1.0 2.0)) #:descending? #t))
    (check-equal? (tensor->list down) '(3.0 2.0 1.0)))

  (test-case "sort still sorts lists as racket/base does"
    (check-equal? (sort '(3 1 2) <) '(1 2 3))
    (check-equal? (sort '("bb" "a" "ccc") > #:key string-length)
                  '("ccc" "bb" "a"))
    (check-equal? (sort '((2 . a) (1 . b)) < #:key car #:cache-keys? #t)
                  '((1 . b) (2 . a)))
    (check-equal? (sort '(3 1 2) < #:key #f) '(1 2 3)))

  (test-case "each form of sort refuses the other's arguments"
    (check-exn exn:fail:contract? (lambda () (sort (tensor '(1.0 2.0)) <)))
    (check-exn exn:fail:contract? (lambda () (sort '(2 1) < #:dim 0)))
    (check-exn #rx"a list is sorted by a less-than\\? procedure"
               (lambda () (sort '(2 1)))))

  (test-case "argsort is the indices half of sort"
    (check-equal? (tensor->list (argsort (tensor '(3.0 1.0 2.0)))) '(1 2 0))
    (check-equal? (tensor->list (argsort (tensor '((3.0 1.0) (0.0 2.0)))
                                         #:dim 0 #:descending? #t))
                  '(0 1 1 0)))

  (test-case "multinomial draws only categories with weight"
    (define draws
      (multinomial (tensor '((0.0 1.0 0.0) (0.5 0.0 0.5))) 8 #:replacement? #t))
    (check-equal? (tensor-shape draws) '(2 8))
    (check-equal? (tensor-dtype draws) 'int64)
    (define flat (tensor->list draws))
    (for ([i (in-range 8)])
      (check-equal? (list-ref flat i) 1)
      (check-not-equal? (list-ref flat (+ 8 i)) 1)))

  (test-case "multinomial replays under a seed and under a generator"
    (define weights (tensor '(0.1 0.2 0.3 0.4)))
    (define (seeded)
      (manual-seed! 11)
      (tensor->list (multinomial weights 16 #:replacement? #t)))
    (check-equal? (seeded) (seeded))
    (define (from-generator)
      (tensor->list (multinomial weights 16 #:replacement? #t
                                 #:generator (make-generator 5))))
    (check-equal? (from-generator) (from-generator)))

  (test-case "a generator draw leaves the global stream where it was"
    (define weights (tensor '(0.25 0.25 0.25 0.25)))
    (manual-seed! 3)
    (define expected (tensor->list (randn 4)))
    (manual-seed! 3)
    (void (multinomial weights 2 #:generator (make-generator 9)))
    (check-equal? (tensor->list (randn 4)) expected))

  (test-case "multinomial refuses a rank-3 tensor and a zero sample count"
    (check-exn #rx"probabilities" (lambda () (multinomial (ones 2 2 2) 1)))
    (check-exn exn:fail:contract? (lambda () (multinomial (ones 3) 0))))

  (test-case "nll-loss over log-softmax is cross-entropy"
    (manual-seed! 0)
    (define logits (randn 4 5))
    (define targets (tensor '(0 4 2 1)))
    (check-= (item (nll-loss (log-softmax logits 1) targets))
             (item (cross-entropy logits targets))
             1e-6))

  (test-case "nll-loss reductions and the ignored index"
    (define log-probs (log (tensor '((0.5 0.5) (0.25 0.75) (0.125 0.875)))))
    (define targets (tensor '(0 1 0)))
    (check-equal? (tensor-shape (nll-loss log-probs targets #:reduction 'none))
                  '(3))
    (check-= (item (nll-loss log-probs targets #:reduction 'sum))
             (- (+ (log 0.5) (log 0.75) (log 0.125)))
             1e-6)
    (check-= (item (nll-loss log-probs (tensor '(0 1 -100))))
             (/ (- (+ (log 0.5) (log 0.75))) 2)
             1e-6)
    (check-exn exn:fail:contract?
               (lambda () (nll-loss log-probs targets #:reduction 'average)))))

#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt")

  (test-case "selection family: take/gather/take-along-dim/where (#73)"
    (define t (reshape (arange 6) 2 3))
    (check-equal? (tensor->list (take t '(0 5 3))) '(0.0 5.0 3.0))
    (check-equal? (tensor->list (take t '#(5 0))) '(5.0 0.0))
    (check-equal? (take '(1 2 3) 2) '(1 2))
    (check-equal? (tensor->list
                   (gather t 1 (to-dtype (tensor '((0 2) (1 0))) 'int64)))
                  '(0.0 2.0 4.0 3.0))
    (check-equal? (tensor->list
                   (take-along-dim t (to-dtype (tensor '((0) (2))) 'int64) 1))
                  '(0.0 5.0))
    (check-equal? (tensor->list
                   (take-along-dim t (to-dtype (tensor '(5 0)) 'int64)))
                  '(5.0 0.0))
    (check-equal? (map tensor->list (where (gt t 2.0)))
                  '((1 1 1) (0 1 2)))
    (check-equal? (tensor->list (where (gt t 2.0) t -1)) '(-1.0 -1.0 -1.0 3.0 4.0 5.0))
    (check-equal? (tensor->list (where (gt t 2.0) t (zeros 2 3)))
                  '(0.0 0.0 0.0 3.0 4.0 5.0))
    (check-equal? (tensor->list (where (gt t 2.0) -1 t))
                  '(0.0 1.0 2.0 -1.0 -1.0 -1.0))
    (let ([it (tensor '(1 2))])
      (check-equal? (tensor->list (where (eq it 1) (add1 (expt 2 53)) it))
                    (list (add1 (expt 2 53)) 2))
      (check-equal? (dtype (where (eq it 1) 7 it)) 'int64))
    (check-equal? (dtype (where (gt t 2.0) 1 0)) 'int64)
    (check-equal? (tensor->list (where (gt t 2.0) 1 0)) '(0 0 0 1 1 1))
    (check-equal? (tensor->list (where (gt t 2.0) 1.5 0.5))
                  '(0.5 0.5 0.5 1.5 1.5 1.5))
    (check-equal? (tensor->list (where (gt t 2.0) (add1 (expt 2 53)) 0))
                  (list 0 0 0 (add1 (expt 2 53)) (add1 (expt 2 53))
                        (add1 (expt 2 53))))
    (check-equal? (tensor->list
                   (index-select t 1 (to-dtype (tensor '(2 0)) 'int64)))
                  '(2.0 0.0 5.0 3.0))
    (check-equal? (tensor->list (masked-select t (gt t 3.0))) '(4.0 5.0))
    (check-equal? (shape (nonzero (gt t 0.0))) '(5 2))
    (let ([it (tensor '((1 2) (3 4)))])
      (check-equal? (dtype (where (eq it 1) it (add1 (expt 2 53)))) 'int64)
      (check-equal? (tensor->list (where (eq it 1) it (add1 (expt 2 53))))
                    (list 1 (add1 (expt 2 53)) (add1 (expt 2 53))
                          (add1 (expt 2 53)))))
    (let ([m (ne (tensor '(1 0 1)) 0)])
      (check-equal? (dtype (where m m 2)) 'int64)
      (check-equal? (tensor->list (where m m (add1 (expt 2 53))))
                    (list 1 (add1 (expt 2 53)) 1)))
    (check-equal? (tensor->list (car (where (gt (tensor 1) 0)))) '(0))
    (check-equal? (tensor->list (car (where (gt (tensor 0) 0)))) '())
    (check-exn exn:fail:contract? (lambda () (take t 2)))
    (check-exn exn:fail:contract? (lambda () (take t (gt t 2.0))))
    (check-equal? (shape (take t '())) '(0))
    (check-exn exn:fail:contract? (lambda () (masked-select t t)))
    (check-exn exn:fail:contract?
               (lambda () (gather t 1 (tensor '((0.5 1.0))))))
    (check-exn exn:fail:contract?
               (lambda () (index-select t 0 (tensor '(0.5)))))
    (check-exn exn:fail:contract?
               (lambda () (index-select t 0 (to-dtype (tensor '((0 1)))
                                                     'int64))))
    (check-exn exn:fail:contract?
               (lambda () (take-along-dim t (tensor '(1.0)))))
    (check-exn exn:fail:contract? (lambda () (where (tensor '(1 0)))))
    (check-exn exn:fail:contract? (lambda () (where (tensor '(1 0)) t 0)))
    (check-exn exn:fail:contract? (lambda () (gather t 1 '(0 1)))))
)

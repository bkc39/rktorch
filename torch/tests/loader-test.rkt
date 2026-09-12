#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../data/loader.rkt")

  (define xs (reshape (arange 12) 6 2))
  (define ys (arange 6 #:dtype 'int64))

  (test-case "generators: seeded streams replay and continue"
    (define a (make-generator 3))
    (define b (make-generator 3))
    (check-true (generator? a))
    (check-false (generator? (ones 1)))
    (define a1 (tensor->list (randperm 16 #:generator a)))
    (define a2 (tensor->list (randperm 16 #:generator a)))
    (check-equal? (sort a1 <) (for/list ([i (in-range 16)]) i))
    (check-false (equal? a1 a2) "the stream continues across draws")
    (check-equal? (tensor->list (randperm 16 #:generator b)) a1)
    (check-equal? (tensor->list (randperm 16 #:generator b)) a2)
    (check-equal? (tensor-dtype (randperm 4)) 'int64)
    (check-equal? (tensor-shape (randperm 0)) '(0))
    (check-exn exn:fail:contract? (lambda () (randperm -1))))

  (test-case "generator draws leave the global stream alone"
    (manual-seed! 5)
    (define expected (tensor->list (randn 4)))
    (manual-seed! 5)
    (void (randperm 64 #:generator (make-generator 9)))
    (check-equal? (tensor->list (randn 4)) expected))

  (test-case "tensor-dataset: length, items, and batch paths"
    (define ds (tensor-dataset xs ys))
    (check-true (dataset? ds))
    (check-true (tensor-dataset? ds))
    (check-equal? (dataset-length ds) 6)
    (define-values (x2 y2) (dataset-ref ds 2))
    (check-equal? (tensor->list x2) '(4.0 5.0))
    (check-equal? (tensor->list y2) '(2))
    (define-values (xr yr) (dataset-batch ds '(1 2 3) default-collate))
    (check-equal? (tensor-shape xr) '(3 2))
    (check-equal? (tensor->list yr) '(1 2 3) "a run narrows")
    (define-values (xg yg) (dataset-batch ds '(5 0 3) default-collate))
    (check-equal? (tensor->list yg) '(5 0 3) "a permutation gathers")
    (check-equal? (tensor->list xg) '(10.0 11.0 0.0 1.0 6.0 7.0))
    (define-values (_xt yt)
      (dataset-batch ds (tensor '(4 1) #:dtype 'int64) default-collate))
    (check-equal? (tensor->list yt) '(4 1) "an index tensor gathers too")
    (check-exn exn:fail:contract? (lambda () (tensor-dataset xs (ones 5))))
    (check-exn #rx"share the first dimension"
               (lambda () (tensor-dataset xs (ones 5 2)))))

  (test-case "a hand-written dataset goes through dataset-ref and collate"
    (struct Squares (n)
      #:methods gen:dataset
      [(define (dataset-length self) (Squares-n self))
       (define (dataset-ref _self i) (values (full (* i i) 2) (tensor i)))])
    (define ds (Squares 5))
    (define-values (xb yb) (dataset-batch ds '(1 3) default-collate))
    (check-equal? (tensor-shape xb) '(2 2))
    (check-equal? (tensor->list xb) '(1.0 1.0 9.0 9.0))
    (check-equal? (tensor->list yb) '(1 3))
    (define loader (dataloader ds #:batch-size 2))
    (check-equal? (dataloader-length loader) 3)
    (check-equal? (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb))
                  '((0 1) (2 3) (4)))
    (define summed
      (dataloader ds #:batch-size 2
                  #:collate (lambda (items)
                              (values (apply + (map (lambda (it) (item (cadr it)))
                                                    items))))))
    (check-equal? (for/list ([s (in-dataloader summed)]) s) '(1 5 4)
                  "a custom collate returns whatever the trainer wants"))

  (test-case "dataloader: sequential batches are narrow views in order"
    (define ds (tensor-dataset xs ys))
    (define loader (dataloader ds #:batch-size 4))
    (check-equal? (dataloader-length loader) 2)
    (check-equal? (for/list ([(xb yb) (in-dataloader loader)])
                    (cons (tensor-shape xb) (tensor->list yb)))
                  '(((4 2) 0 1 2 3) ((2 2) 4 5)))
    (check-equal? (dataloader-length (dataloader ds #:batch-size 4 #:drop-last? #t))
                  1)
    (check-equal? (for/list ([(_xb yb) (in-dataloader
                                        (dataloader ds #:batch-size 4
                                                    #:drop-last? #t))])
                    (tensor->list yb))
                  '((0 1 2 3)))
    ;; full batch, no shuffle: exactly the dataset's tensors, as the oracles use
    (define full
      (for/first ([(xb yb) (in-dataloader (dataloader ds #:batch-size 6))])
        (list xb yb)))
    (check-equal? (tensor->list (car full)) (tensor->list xs))
    (check-equal? (tensor->list (cadr full)) (tensor->list ys)))

  (test-case "dataloader: shuffle draws a permutation per traversal"
    (define ds (tensor-dataset xs ys))
    (define (orders g)
      (define loader
        (dataloader ds #:batch-size 4 #:shuffle? #t #:generator g))
      (for/list ([_e (in-range 2)])
        (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb))))
    (define first-run (orders (make-generator 1)))
    (check-equal? first-run (orders (make-generator 1)) "seeded replay")
    (for ([epoch (in-list first-run)])
      (check-equal? (sort (apply append epoch) <) '(0 1 2 3 4 5)))
    (check-false (equal? (car first-run) (cadr first-run))
                 "the second epoch is a fresh draw from the same stream")
    ;; one epoch draws what a DataLoader epoch draws: a seed word, then the
    ;; permutation, then the sampler's discarded remainder permutation
    (define g (make-generator 1))
    (void (draw-seed #:generator g))
    (define expected (tensor->list (randperm 6 #:generator g)))
    (define loader
      (dataloader ds #:batch-size 6 #:shuffle? #t #:generator (make-generator 1)))
    (define batch
      (for/first ([(xb yb) (in-dataloader loader)]) (list xb yb)))
    (check-equal? (tensor->list (cadr batch)) expected
                  "the batch is the permutation")
    (check-equal? (tensor->list (car batch))
                  (apply append (for/list ([i (in-list expected)])
                                  (tensor->list (select xs 0 i))))))

  (test-case "in-epochs numbers epochs and continues the stream"
    (define ds (tensor-dataset xs ys))
    (define loader
      (dataloader ds #:batch-size 4 #:shuffle? #t #:generator (make-generator 2)))
    (define seen
      (for/list ([(epoch _xb yb) (in-epochs loader 3)])
        (cons epoch (tensor->list yb))))
    (check-equal? (map car seen) '(0 0 1 1 2 2))
    (define by-epoch
      (for/list ([e (in-range 3)])
        (apply append (for/list ([s (in-list seen)] #:when (= (car s) e))
                        (cdr s)))))
    (define twin
      (dataloader ds #:batch-size 4 #:shuffle? #t #:generator (make-generator 2)))
    (check-equal? by-epoch
                  (for/list ([_e (in-range 3)])
                    (apply append
                           (for/list ([(_xb yb) (in-dataloader twin)])
                             (tensor->list yb))))
                  "in-epochs and repeated in-dataloader traversals agree")
    (check-equal? (for/list ([(epoch _xb _yb) (in-epochs loader 0)]) epoch)
                  '())))

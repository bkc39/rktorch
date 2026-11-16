#lang racket/base

(module+ test
  (require (only-in racket/list filter-not make-list)
           rackunit
           (only-in "../data/loader.rkt"
                    dataloader dataloader-length in-dataloader tensor-dataset)
           (only-in "../main.rkt" make-generator ones tensor tensor->list)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../train.rkt"
                    define-init define-optimizers define-test-step define-trainer
                    define-training-step define-validation-step fit test))

  (define n 16)
  (define batch 4)
  (define epochs 3)

  (define (shuffled #:workers [workers 0])
    (dataloader (tensor-dataset (ones n 1) (tensor (for/list ([i (in-range n)]) i)))
                #:batch-size batch #:shuffle? #t #:generator (make-generator 0)
                #:workers workers))

  (define (record! order i)
    (set-box! order (append (unbox order) (tensor->list i))))

  (define-trainer Seen (net order)
    (define-init (sink) (set! net (Linear 1 1)) (set! order sink))
    (define-optimizers (sgd (parameters net) #:lr 0.0))
    (define-training-step (x i)
      (record! order i)
      (mse-loss (net x) x))
    (define-validation-step (x i)
      (record! order i)
      (mse-loss (net x) x))
    (define-test-step (x i)
      (record! order i)
      (mse-loss (net x) x)))

  (define (fetched-by-epoch loader)
    (for*/list ([_ (in-range epochs)]
                [i (in-list (for/list ([_ (in-range (dataloader-length loader))]
                                       [(_x i) (in-dataloader loader)])
                              i))]
                [v (in-list (tensor->list i))])
      v))

  (define (exhausted loader)
    (for*/list ([_ (in-range epochs)]
                [(_x i) (in-dataloader loader)]
                [v (in-list (tensor->list i))])
      v))

  (define (trained-order loader)
    (define order (box '()))
    (fit (Seen order) loader #:max-epochs epochs #:logger #f)
    (unbox order))

  (test-case "fit pulls a sized loader no further than its last batch, as Lightning's fetcher"
    (check-equal? (trained-order (shuffled)) (fetched-by-epoch (shuffled)))
    (check-not-equal? (fetched-by-epoch (shuffled)) (exhausted (shuffled))
                      "the pull past the end draws torch's trailing permutation"))

  (test-case "with workers the prefetch reaches the end, as torch's multi-process loader"
    (check-equal? (trained-order (shuffled #:workers 2)) (exhausted (shuffled))))

  (test-case "validation pulls a shuffled loader no further than its last batch"
    (define order (box '()))
    (define unrecorded
      (dataloader (tensor-dataset (ones batch 1) (tensor (make-list batch -1))) #:batch-size batch))
    (fit (Seen order) unrecorded #:val (shuffled) #:num-sanity-val-steps 0 #:max-epochs epochs
         #:logger #f)
    (check-equal? (filter-not negative? (unbox order)) (fetched-by-epoch (shuffled))))

  (test-case "test pulls a shuffled loader no further than its last batch"
    (define order (box '()))
    (define seen (Seen order))
    (define loader (shuffled))
    (for ([_ (in-range epochs)])
      (test seen loader #:logger #f))
    (check-equal? (unbox order) (fetched-by-epoch (shuffled)))))

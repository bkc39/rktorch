#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt"
                    dataloader dataloader-length in-dataloader tensor-dataset)
           (only-in "../main.rkt" make-generator ones tensor tensor->list)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../train.rkt"
                    define-init define-optimizers define-trainer define-training-step fit))

  (define n 16)
  (define batch 4)
  (define epochs 3)

  (define (shuffled)
    (dataloader (tensor-dataset (ones n 1) (tensor (for/list ([i (in-range n)]) i)))
                #:batch-size batch #:shuffle? #t #:generator (make-generator 0)))

  (define-trainer Seen (net order)
    (define-init (sink) (set! net (Linear 1 1)) (set! order sink))
    (define-optimizers (sgd (parameters net) #:lr 0.0))
    (define-training-step (x i)
      (set-box! order (append (unbox order) (tensor->list i)))
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

  (test-case "fit pulls a sized loader no further than its last batch, as Lightning's fetcher"
    (define order (box '()))
    (fit (Seen order) (shuffled) #:max-epochs epochs #:logger #f)
    (check-equal? (unbox order) (fetched-by-epoch (shuffled)))
    (check-not-equal? (fetched-by-epoch (shuffled)) (exhausted (shuffled))
                      "the pull past the end draws torch's trailing permutation")))

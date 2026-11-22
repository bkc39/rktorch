#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" make-generator manual-seed! randn)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt" collecting-logger collecting-steps))

  (define-trainer Logging (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (log! 'train-loss loss)
      (log! 'train-mean loss #:on-step? #f #:on-epoch? #t)
      (when (= (run-batch-index run) 3)
        (log! 'rare loss))
      loss)
    (define-validation-step (x y)
      (log! 'val-loss (mse-loss (net x) y))))

  (define-callback epoch-end-saver (dir drop-book?)
    #:kind checkpoint
    (define-init (d drop?) (set! dir d) (set! drop-book? drop?))
    (define-hook (on-save-checkpoint tree)
      (when drop-book?
        (hash-remove! tree "metric_book")))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (build-path dir (format "epoch=~a.ckpt" (run-epoch run))))))

  (define (data n seed)
    (define g (make-generator seed))
    (tensor-dataset (randn n 2 #:generator g) (randn n 1 #:generator g)))

  (define (epoch-start-reader seen)
    (lambda-callback
     #:on-train-epoch-start
     (lambda (r)
       (set-box! seen (cons (list (run-epoch r)
                                  (for/list ([key (in-list '(train-loss train-mean rare val-loss))])
                                    (run-metric r key))
                                  (run-metrics r))
                            (unbox seen))))))

  (define (go dir max-epochs ckpt #:drop-book? [drop-book? #f])
    (manual-seed! 0)
    (define seen (box '()))
    (define l (collecting-logger))
    (fit (Logging) (dataloader (data 8 1) #:batch-size 2)
         #:val (dataloader (data 6 2) #:batch-size 2)
         #:max-epochs max-epochs #:logger l #:log-every-n-steps 3 #:ckpt-path ckpt
         #:callbacks (list (epoch-start-reader seen) (epoch-end-saver dir drop-book?)))
    (values (reverse (unbox seen)) (collecting-steps l)))

  (define (straight-and-split dir #:drop-book? [drop-book? #f])
    (define-values (straight-reads straight-rows) (go (build-path dir "straight") 4 #f))
    (define split (build-path dir "split"))
    (define-values (first-reads first-rows) (go split 2 #f #:drop-book? drop-book?))
    (define-values (later-reads later-rows) (go split 4 (build-path split "epoch=1.ckpt")))
    (values straight-reads (append first-reads later-reads)
            straight-rows (append first-rows later-rows)))

  (define (with-legs proc)
    (with-temporary-directory (dir #:template "book-~a")
      (make-directory (build-path dir "straight"))
      (make-directory (build-path dir "split"))
      (proc dir)))

  (test-case "after an epoch-end resume run-metric reads what the straight run reads"
    (with-legs
     (lambda (dir)
       (define-values (straight split _straight-rows _split-rows) (straight-and-split dir))
       (check-equal? split straight))))

  (test-case "a step value pending at the epoch-end save is published in the resumed run's row"
    (with-legs
     (lambda (dir)
       (define-values (_straight _split straight-rows split-rows) (straight-and-split dir))
       (check-equal? (map car split-rows) (map car straight-rows) "the row steps")
       (check-equal? split-rows straight-rows))))

  (test-case "a checkpoint without a metric_book entry resumes from its callback_metrics"
    (with-legs
     (lambda (dir)
       (define-values (_straight split _straight-rows _split-rows)
         (straight-and-split dir #:drop-book? #t))
       (define ckpt (read-checkpoint (build-path dir "split" "epoch=1.ckpt")))
       (check-false (hash-has-key? ckpt "metric_book"))
       (match-define (list epoch _named table) (list-ref split 2))
       (check-equal? epoch 2)
       (check-equal? table
                     (for/hasheq ([(k v) (in-hash (hash-ref ckpt "callback_metrics"))])
                       (values (string->symbol k) v))
                     "the run-metrics table is the saved callback_metrics")))))

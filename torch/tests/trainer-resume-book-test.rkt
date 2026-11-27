#lang racket/base

(module+ test
  (require (only-in racket/list last)
           (only-in racket/match match-define match-lambda)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in threading ~>)
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" make-generator manual-seed! randn)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt"
                    call-with-logged-warnings collecting-logger collecting-steps))

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
      (log! 'val-loss (mse-loss (net x) y)))
    (define-hook (on-train-epoch-end)
      (log! 'epoch-tag (exact->inexact (run-epoch run)))))

  (define-callback epoch-end-saver (dir drop-book?)
    #:kind checkpoint
    (define-init (d drop?) (set! dir d) (set! drop-book? drop?))
    (define-hook (on-save-checkpoint tree)
      (when drop-book?
        (hash-remove! tree "metric_book")))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (epoch-path dir run))))

  (define (epoch-path dir r)
    (build-path dir (format "epoch=~a.ckpt" (run-epoch r))))

  (define (plain-saver dir)
    (lambda-callback #:on-train-epoch-end (lambda (r) (save-checkpoint! r (epoch-path dir r)))))

  (define (data n seed)
    (define g (make-generator seed))
    (tensor-dataset (randn n 2 #:generator g) (randn n 1 #:generator g)))

  (define logged-keys '(train-loss train-mean rare val-loss epoch-tag))

  (define (epoch-start-reader seen)
    (lambda-callback
     #:on-train-epoch-start
     (lambda (r)
       (set-box! seen (cons (list (run-epoch r)
                                  (for/list ([key (in-list logged-keys)]) (run-metric r key))
                                  (run-metrics r))
                            (unbox seen))))))

  (define (go dir max-epochs ckpt saver)
    (manual-seed! 0)
    (define seen (box '()))
    (define l (collecting-logger))
    (fit (Logging) (dataloader (data 8 1) #:batch-size 2)
         #:val (dataloader (data 6 2) #:batch-size 2)
         #:max-epochs max-epochs #:logger l #:log-every-n-steps 3 #:ckpt-path ckpt
         #:callbacks (list (epoch-start-reader seen) (saver dir)))
    (list (reverse (unbox seen)) (collecting-steps l)))

  (define (book-saver dir) (epoch-end-saver dir #f))

  (define (straight-and-split dir #:drop-book? [drop-book? #f] #:saver [saver book-saver])
    (match-define (list straight-reads straight-rows) (go (build-path dir "straight") 4 #f saver))
    (define split (build-path dir "split"))
    (define first-saver (if drop-book? (lambda (d) (epoch-end-saver d #t)) saver))
    (match-define (list first-reads first-rows) (go split 2 #f first-saver))
    (define-values (later warnings)
      (call-with-logged-warnings
       (lambda ()
         (parameterize ([current-error-port (open-output-nowhere)])
           (go split 4 (build-path split "epoch=1.ckpt") saver)))))
    (match-define (list later-reads later-rows) later)
    (values straight-reads (append first-reads later-reads)
            straight-rows (append first-rows later-rows)
            warnings))

  (define (with-legs proc)
    (with-temporary-directory (dir #:template "book-~a")
      (make-directory (build-path dir "straight"))
      (make-directory (build-path dir "split"))
      (proc dir)))

  (test-case "after an epoch-end resume run-metric reads what the straight run reads"
    (with-legs
     (lambda (dir)
       (define-values (straight split _straight-rows _split-rows warnings) (straight-and-split dir))
       (check-equal? split straight)
       (check-equal? warnings '() "a checkpoint-kind save resumes without a warning")
       (check-true (~> (build-path dir "split" "epoch=1.ckpt") read-checkpoint (hash-ref "loops")
                       (hash-ref "fit_loop") (hash-ref "epoch_end_logged"))))))

  (test-case "a step value pending at the epoch-end save is published in the resumed run's row"
    (with-legs
     (lambda (dir)
       (define-values (_straight _split straight-rows split-rows _warnings)
         (straight-and-split dir))
       (check-equal? (map car split-rows) (map car straight-rows) "the row steps")
       (check-equal? split-rows straight-rows))))

  (test-case "a checkpoint without a metric_book entry resumes from its callback_metrics"
    (with-legs
     (lambda (dir)
       (define-values (_straight split _straight-rows _split-rows _warnings)
         (straight-and-split dir #:drop-book? #t))
       (define ckpt (read-checkpoint (build-path dir "split" "epoch=1.ckpt")))
       (check-false (hash-has-key? ckpt "metric_book"))
       (match-define (list epoch _named table) (list-ref split 2))
       (check-equal? epoch 2)
       (check-equal? table
                     (for/hasheq ([(k v) (in-hash (hash-ref ckpt "callback_metrics"))])
                       (values (string->symbol k) v))
                     "the run-metrics table is the saved callback_metrics"))))

  (test-case "a plain saver's epoch-end file warns on resume and reads stale epoch-end logs"
    (with-legs
     (lambda (dir)
       (define-values (straight split straight-rows split-rows warnings)
         (straight-and-split dir #:saver plain-saver))
       (define path (path->string (build-path dir "split" "epoch=1.ckpt")))
       (check-equal? (length warnings) 1)
       (check-regexp-match (regexp-quote path) (car warnings))
       (check-regexp-match #rx"before or during the trainer's on-train-epoch-end" (car warnings))
       (check-false (~> (read-checkpoint path) (hash-ref "loops") (hash-ref "fit_loop")
                        (hash-ref "epoch_end_logged")))
       (define (epoch-tags reads) (map (match-lambda [(list _ named _) (last named)]) reads))
       (check-equal? (epoch-tags straight) '(#f 0.0 1.0 2.0))
       (check-equal? (epoch-tags split) '(#f 0.0 0.0 2.0)
                     "the resumed epoch reads the value the trainer logged an epoch earlier")
       (check-equal? split-rows straight-rows))))

  (define odd-key (string->uninterned-symbol "odd"))

  (define-trainer Odd (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (log! odd-key loss)
      loss))

  (test-case "a value logged under an uninterned symbol is saved under its name"
    (with-temporary-directory (dir #:template "odd-~a")
      (define path (build-path dir "odd.ckpt"))
      (void (fit (Odd) (dataloader (data 4 1) #:batch-size 2) #:max-epochs 1 #:logger #f
                 #:callbacks (list (lambda-callback
                                    #:on-train-end (lambda (r) (save-checkpoint! r path))))))
      (define train (~> path read-checkpoint (hash-ref "metric_book") (hash-ref "train")))
      (check-true (for/or ([table (in-hash-values train)]) (hash-has-key? table "odd")))))

  (define-trainer Twins (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (log! 'odd loss)
      (log! odd-key loss)
      loss))

  (test-case "two logged names spelled the same are refused at save, naming both"
    (with-temporary-directory (dir #:template "twins-~a")
      (define path (build-path dir "twins.ckpt"))
      (check-exn (lambda (e)
                   (and (exn:fail:contract? e)
                        (regexp-match? #rx"spelled the same" (exn-message e))
                        (regexp-match? #rx"key: \"metric_book[.]train[.]steps\"" (exn-message e))
                        (regexp-match? #rx"names: '[(]odd odd[)]" (exn-message e))))
                 (lambda ()
                   (fit (Twins) (dataloader (data 4 1) #:batch-size 2) #:max-epochs 1 #:logger #f
                        #:callbacks (list (lambda-callback
                                           #:on-train-end (lambda (r) (save-checkpoint! r path)))))))
      (check-false (file-exists? path)))))


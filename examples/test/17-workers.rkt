#lang racket/base

;; Runner + tests for the literate torch/scribblings/examples/17-workers.rkt.

(require (only-in racket/match match-define)
         (only-in racket/math nan?)
         torch
         torch/data/loader
         torch/scribblings/examples/17-workers
         (only-in torch/scribblings/examples/05-mnist pick-device)
         (only-in torch/vision/mnist load-mnist-fixture))

(module+ main
  (require (only-in "private/env.rkt" env-number))
  (define workers (env-number "WORKERS" 4))
  (printf "device: ~a\n" (pick-device))
  (define-values (xs ys) (load-mnist-fixture))
  (define ds (encoded-digits xs ys))
  (printf "same first batch for workers 0 and ~a: ~a\n" workers
          (equal? (first-labels ds 0) (first-labels ds workers)))
  (define start (current-inexact-milliseconds))
  (for ([i (in-range (length ds))])
    (dataset-ref ds i))
  (printf "decode + jitter: ~a ms per image\n"
          (real->decimal-string (/ (- (current-inexact-milliseconds) start) (length ds)) 3))
  (for ([row (in-list (train-workers #:workers workers))])
    (match-define (list n loss seconds acc) row)
    (printf "workers ~a: ~a s, mean loss ~a, test acc ~a\n"
            n (real->decimal-string seconds 2) (real->decimal-string loss 4)
            (real->decimal-string acc 4))))

(module+ test
  (require rackunit)
  (define rows (run-example #:workers 2 #:device 'cpu))
  (check-equal? (map car rows) '(0 2))
  (for ([row (in-list rows)])
    (match-define (list n loss seconds acc) row)
    (check-true (and (rational? loss) (not (nan? loss)))
                (format "workers ~a: non-finite loss ~a" n loss))
    (check-true (positive? seconds))
    (check-true (and (> acc 0.1) (<= acc 1.0))
                (format "workers ~a: accuracy out of range ~a" n acc)))
  (check-true (< (cadr (cadr rows)) (cadr (car rows)))
              (format "the second epoch's loss did not fall: ~a" rows))

  (define-values (xs ys) (load-mnist-fixture))
  (define ds (encoded-digits xs ys))
  (check-equal? (length ds) 256)
  (check-equal? (first-labels ds 0) (first-labels ds 2)
                "the batch order depends on the worker count")
  (define (first-images workers)
    (for/first ([(xb _yb) (in-dataloader
                           (dataloader ds #:batch-size 8 #:shuffle? #t
                                       #:collate collate-digits
                                       #:generator (make-generator 0)
                                       #:workers workers))])
      (check-equal? (shape xb) '(8 1 28 28))
      (check-equal? (dtype xb) 'float32)
      (tensor->list xb)))
  (check-equal? (first-images 2) (first-images 2)
                "a seeded loader did not replay its jitter")
  (define plain
    (for/first ([(xb _yb) (in-dataloader
                           (dataloader (tensor-dataset xs ys)
                                       #:batch-size 8 #:shuffle? #t
                                       #:generator (make-generator 0)))])
      xb))
  (define jittered
    (for/first ([(xb _yb) (in-dataloader
                           (dataloader ds #:batch-size 8 #:shuffle? #t
                                       #:collate collate-digits
                                       #:generator (make-generator 0)))])
      xb))
  (check-not-equal? (tensor->list jittered) (tensor->list plain)
                    "the jitter left every image in place")
  (check-true (< (item (mean (abs (- jittered plain)))) 0.2)
              "the decoded images are not the encoded ones, moved"))

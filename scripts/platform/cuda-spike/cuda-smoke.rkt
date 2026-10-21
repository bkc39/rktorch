#lang racket/base
(require (prefix-in t: torch))

(define (flatten-nested v)
  (if (list? v) (apply append (map flatten-nested v)) (list v)))

(define (max-abs-difference on-gpu on-cpu)
  (for/fold ([worst 0.0])
            ([x (in-list (flatten-nested (t:tensor->list (t:to on-gpu 'cpu))))]
             [y (in-list (flatten-nested (t:tensor->list on-cpu)))])
    (max worst (abs (- x y)))))

(define gpu (t:cuda-device))

(printf "cuda-available? ~a\n" (t:cuda-available?))
(printf "cuda-device-count ~a\n" (t:cuda-device-count))

(t:manual-seed! 0)
(define a (t:randn 64 32))
(define b (t:randn 32 16))
(define product (t:matmul (t:to a gpu) (t:to b gpu)))
(printf "matmul on ~a, max |cuda - cpu| ~a\n"
        (t:device-type (t:device product))
        (max-abs-difference product (t:matmul a b)))

(define images (t:randn 2 3 16 16))
(define kernels (t:randn 4 3 3 3))
(define convolved (t:conv2d (t:to images gpu) (t:to kernels gpu)))
(printf "conv2d on ~a, max |cuda - cpu| ~a\n"
        (t:device-type (t:device convolved))
        (max-abs-difference convolved (t:conv2d images kernels)))

(printf "cuda-memory-stats ~a\n" (t:cuda-memory-stats gpu))

(define run-example (dynamic-require 'torch/scribblings/examples/04-mlp 'run-example))
(define-values (losses _net used) (run-example gpu))
(printf "04-mlp on ~a\n" (t:device-type used))
(for ([loss (in-list losses)] [step (in-naturals 1)])
  (printf "  step ~a  loss ~a\n" step loss))

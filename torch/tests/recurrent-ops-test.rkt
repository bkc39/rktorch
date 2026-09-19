#lang racket/base

(module+ test
  (require rackunit
           (only-in "../generated.rkt" gru-input lstm-input)
           "../main.rkt")

  (define (gru-weights input hidden)
    (list (randn (* 3 hidden) input) (randn (* 3 hidden) hidden)
          (randn (* 3 hidden)) (randn (* 3 hidden))))

  (define (lstm-weights input hidden)
    (list (randn (* 4 hidden) input) (randn (* 4 hidden) hidden)
          (randn (* 4 hidden)) (randn (* 4 hidden))))

  (define (close? a b)
    (for/and ([x (in-list (tensor->list a))]
              [y (in-list (tensor->list b))])
      (< (abs (- x y)) 1e-4)))

  (test-case "lstm-input answers output, h_n and c_n"
    (manual-seed! 0)
    (define-values (output h-n c-n)
      (lstm-input (randn 5 2 3) (list (zeros 1 2 4) (zeros 1 2 4))
                  (lstm-weights 3 4) #t 1 0.0 #f #f #f))
    (check-equal? (tensor-shape output) '(5 2 4))
    (check-equal? (tensor-shape h-n) '(1 2 4))
    (check-equal? (tensor-shape c-n) '(1 2 4))
    (check-true (close? (select output 0 4) (select h-n 0 0))))

  (test-case "gru-input is batch-first on request and bidirectional"
    (manual-seed! 0)
    (define-values (output h-n)
      (gru-input (randn 2 5 3) (zeros 2 2 4)
                 (append (gru-weights 3 4) (gru-weights 3 4))
                 #t 1 0.0 #f #t #t))
    (check-equal? (tensor-shape output) '(2 5 8))
    (check-equal? (tensor-shape h-n) '(2 2 4)))

  (test-case "gradients reach the flat weights and the initial state"
    (manual-seed! 0)
    (define weights (map requires-grad! (gru-weights 3 4)))
    (define h0 (requires-grad! (randn 1 2 4)))
    (define-values (output _h-n)
      (gru-input (randn 5 2 3) h0 weights #t 1 0.0 #f #f #f))
    (backward! (sum output))
    (for ([w (in-list weights)])
      (check-equal? (tensor-shape (grad w)) (tensor-shape w)))
    (check-equal? (tensor-shape (grad h0)) '(1 2 4)))

  (test-case "a weight list of the wrong length is refused by ATen"
    (check-exn exn:fail?
               (lambda ()
                 (gru-input (randn 5 2 3) (zeros 1 2 4)
                            (list (randn 12 3)) #t 1 0.0 #f #f #f))))

  (when (cuda-available?)
    (test-case "the cudnn recurrences agree with the CPU ones"
      (manual-seed! 0)
      (define input (randn 5 2 3))
      (define h0 (randn 1 2 4))
      (define c0 (randn 1 2 4))
      (define lstm-ws (lstm-weights 3 4))
      (define gru-ws (gru-weights 3 4))
      (define (on-gpu t) (to-device t 'cuda))
      (define-values (cpu-out cpu-h cpu-c)
        (lstm-input input (list h0 c0) lstm-ws #t 1 0.0 #f #f #f))
      (define-values (gpu-out gpu-h gpu-c)
        (lstm-input (on-gpu input) (map on-gpu (list h0 c0))
                    (map on-gpu lstm-ws) #t 1 0.0 #f #f #f))
      (check-equal? (device-type (tensor-device gpu-out)) 'cuda)
      (check-true (close? cpu-out (to-device gpu-out 'cpu)))
      (check-true (close? cpu-h (to-device gpu-h 'cpu)))
      (check-true (close? cpu-c (to-device gpu-c 'cpu)))
      (define-values (cpu-gru _cpu-gru-h)
        (gru-input input h0 gru-ws #t 1 0.0 #f #f #f))
      (define-values (gpu-gru _gpu-gru-h)
        (gru-input (on-gpu input) (on-gpu h0) (map on-gpu gru-ws)
                   #t 1 0.0 #f #f #f))
      (check-true (close? cpu-gru (to-device gpu-gru 'cpu))))

    (test-case "every output of a CUDA call is accounted to the device"
      (define (cuda-bytes)
        (for/sum ([entry (in-list (native-memory-use))]
                  #:when (eq? (device-type (car entry)) 'cuda))
          (cdr entry)))
      (define scores (to-device (randn 64) 'cuda))
      (define before (cuda-bytes))
      (define-values (top indices) (topk scores 16))
      (check-equal? (- (cuda-bytes) before) (+ (* 16 4) (* 16 8)))
      (check-equal? (device-type (tensor-device indices)) 'cuda)
      (check-equal? (tensor-shape top) '(16)))

    (test-case "multinomial samples on the device"
      (define draws
        (multinomial (to-device (tensor '(0.0 1.0 0.0)) 'cuda) 4
                     #:replacement? #t))
      (check-equal? (device-type (tensor-device draws)) 'cuda)
      (check-equal? (tensor->list (to-device draws 'cpu)) '(1 1 1 1)))))

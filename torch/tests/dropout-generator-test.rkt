#lang racket/base

(module+ test
  (require rackunit
           (only-in threading ~>)
           (prefix-in g: (only-in "../generated.rkt" dropout))
           (only-in "../main.rkt"
                    * / < arange backward! current-generator draw-seed dtype grad
                    make-generator manual-seed! ones rand-like reshape sum tensor
                    tensor->list to-dtype)
           (only-in "../nn.rkt" Dropout dropout eval!)
           (only-in "private/blame.rkt" blamed contract-violation?)
           (only-in "private/python-env.rkt" python-check python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define caller (blamed "dropout-generator-test.rkt"))

  (define (x) (~> (arange 1 257) (reshape 4 64)))

  (define (functional g) (tensor->list (dropout (x) #:p 0.3 #:generator g)))

  (define (layer g) (tensor->list ((Dropout #:p 0.3) (x) #:generator g)))

  (define (through-parameter draw seed)
    (parameterize ([current-generator (make-generator seed)])
      (draw #f)))

  (define (unchanged? g seed)
    (equal? (draw-seed #:generator g) (draw-seed #:generator (make-generator seed))))

  (bounded-test "one seed through current-generator draws one mask"
    (for ([draw (list functional layer)] [name '(dropout Dropout)])
      (check-equal? (through-parameter draw 3) (through-parameter draw 3)
                    (format "~a" name))
      (check-equal? (through-parameter draw 3) (draw (make-generator 3))
                    (format "~a: #:generator and current-generator agree" name))))

  (bounded-test "different seeds draw different masks"
    (for ([draw (list functional layer)] [name '(dropout Dropout)])
      (check-not-equal? (through-parameter draw 3) (through-parameter draw 4)
                        (format "~a" name))
      (check-not-equal? (draw (make-generator 3)) (draw (make-generator 4))
                        (format "~a, passed" name))))

  (bounded-test "the functional form and the layer draw the same mask"
    (check-equal? (functional (make-generator 9)) (layer (make-generator 9))))

  (bounded-test "an explicit #:generator wins over current-generator"
    (define parameterized (make-generator 1))
    (check-equal? (parameterize ([current-generator parameterized])
                    (functional (make-generator 2)))
                  (functional (make-generator 2)))
    (check-true (unchanged? parameterized 1)))

  (bounded-test "the mask is rand-like below 1 - p, the kept scaled by 1/(1 - p)"
    (define noise
      (~> (rand-like (x) #:generator (make-generator 5)) (< 0.7) (to-dtype 'float32) (/ 0.7)))
    (check-equal? (functional (make-generator 5)) (tensor->list (* (x) noise))))

  (bounded-test "an evaluating layer is the identity and draws nothing"
    (define g (make-generator 6))
    (define d (eval! (Dropout #:p 0.5)))
    (define in (tensor->list (x)))
    (check-equal? (tensor->list (d (x) #:generator g)) in)
    (check-equal? (tensor->list (parameterize ([current-generator g]) (d (x)))) in)
    (check-true (unchanged? g 6)))

  (bounded-test "#:training? #f and #:p 0 draw nothing"
    (define g (make-generator 8))
    (define in (tensor->list (x)))
    (check-equal? (tensor->list (dropout (x) #:training? #f #:generator g)) in)
    (check-equal? (tensor->list (dropout (x) #:p 0 #:generator g)) in)
    (check-true (unchanged? g 8)))

  (bounded-test "without a generator dropout is libtorch's, on the global stream"
    (manual-seed! 11)
    (define ours (tensor->list (dropout (x) #:p 0.3)))
    (define layer-drawn (begin (manual-seed! 11) (tensor->list ((Dropout #:p 0.3) (x)))))
    (manual-seed! 11)
    (check-equal? ours (tensor->list (g:dropout (x) 0.3 #t)))
    (check-equal? layer-drawn ours))

  (bounded-test "the gradient through a generator's mask is the mask"
    (define in (ones 4 64 #:requires-grad? #t))
    (define out (dropout in #:p 0.3 #:generator (make-generator 5)))
    (backward! (sum out))
    (check-equal? (tensor->list (grad in)) (tensor->list out)))

  (bounded-test "a contract violation blames the caller"
    (check-exn (contract-violation? caller #rx"generator")
               (lambda () (dropout (x) #:generator 5)))
    (check-exn (contract-violation? caller #rx"floating-point")
               (lambda () (dropout (tensor '(1 2 3) #:dtype 'int64)
                                   #:generator (make-generator 1))))
    (check-exn (contract-violation? caller #rx"floating-point")
               (lambda ()
                 (parameterize ([current-generator (make-generator 1)])
                   (dropout (tensor '(1 2 3) #:dtype 'int64))))))

  (bounded-test "an integer tensor passes untouched without a draw"
    (define ints (tensor '(1 2 3) #:dtype 'int64))
    (check-equal? (tensor->list (dropout ints #:training? #f #:generator (make-generator 1)))
                  '(1 2 3))
    (check-equal? (dtype (dropout ints #:training? #f)) 'int64))

  (bounded-test "the layer's application blames its caller"
    (check-exn #rx"Dropout: contract violation.*generator"
               (lambda () ((Dropout) (x) #:generator 'g))))

  (cond
    [(python-torch-available?)
     (bounded-test "masks match PyTorch's, with and without a generator"
       (define j (python-check "dropout_generator.py"))
       (manual-seed! 7)
       (check-equal? (tensor->list (dropout (x) #:p 0.3)) (hash-ref j 'global)
                     "F.dropout after torch.manual_seed")
       (check-equal? (functional (make-generator 7)) (hash-ref j 'generator)
                     "torch.Generator().manual_seed(7) through torch.rand"))]
    [else
     (printf "[dropout-generator-test] skipped parity: python3 `torch` not available ~a\n"
             "(run inside the default `nix develop`)")]))

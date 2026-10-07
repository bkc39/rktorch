#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt")

  (define (close? a b [eps 1e-5])
    (and (equal? (shape a) (shape b))
         (for/and ([x (in-list (tensor->list a))]
                   [y (in-list (tensor->list b))])
           (< (abs (- x y)) eps))))

  (define (attention-by-hand q k v #:scale [scale #f] #:bias [bias #f])
    (define e (car (reverse (shape q))))
    (define scores (* (@ q (transpose k -2 -1)) (or scale (/ 1 (sqrt e)))))
    (@ (softmax (if bias (+ scores bias) scores) -1) v))

  (define (hidden->bias hidden)
    (masked-fill (zeros-like hidden #:dtype 'float32) hidden -inf.0))

  (define ((message-matching pattern) e)
    (and (exn:fail:contract? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (manual-seed! 0)
  (define q (randn 2 3 4))
  (define k (randn 2 5 4))
  (define v (randn 2 5 6))

  (test-case "softmax(q k^T / sqrt E) v over [... L E] [... S E] [... S Ev]"
    (define out (scaled-dot-product-attention q k v))
    (check-equal? (shape out) '(2 3 6))
    (check-true (close? out (attention-by-hand q k v)))
    (define heads (randn 2 4 3 8))
    (check-equal? (shape (scaled-dot-product-attention heads heads heads))
                  '(2 4 3 8))
    (check-equal? (shape (scaled-dot-product-attention (randn 3 4) (randn 5 4)
                                                       (randn 5 2)))
                  '(3 2)))

  (test-case "#:scale replaces 1/sqrt E; zero averages the values"
    (check-true (close? (scaled-dot-product-attention q k v #:scale 1/2)
                        (scaled-dot-product-attention q k v)))
    (check-true (close? (scaled-dot-product-attention q k v #:scale 2)
                        (attention-by-hand q k v #:scale 2)))
    (check-true (close? (scaled-dot-product-attention q k v #:scale 0)
                        (@ (full 0.2 2 3 5) v))))

  (test-case "a boolean mask is #t where a query may attend"
    (define lower (tril (ones 3 5 #:dtype 'bool)))
    (define upper (eq (tril (ones 3 5)) 0))
    (check-true (close? (scaled-dot-product-attention q k v #:mask lower)
                        (attention-by-hand q k v
                                           #:bias (hidden->bias upper))))
    (define first-key (eq (tensor '(1 0 0 0 0)) 1))
    (define out (scaled-dot-product-attention q k v #:mask first-key))
    (for ([i (in-range 3)])
      (check-true (close? (select out 1 i) (select v 1 0))))
    (check-true (close? (scaled-dot-product-attention
                         q k v #:mask (zeros 5 #:dtype 'bool))
                        (zeros 2 3 6))))

  (test-case "a float mask is added to the scores"
    (define bias (randn 3 5))
    (check-true (close? (scaled-dot-product-attention q k v #:mask bias)
                        (attention-by-hand q k v #:bias bias))))

  (test-case "#:causal? hides every later key, as the masked-fill idiom does"
    (define square (randn 2 5 4))
    (define causal (scaled-dot-product-attention square square v #:causal? #t))
    (define later (eq (tril (ones 5 5)) 0))
    (check-true (close? causal
                        (attention-by-hand square square v
                                           #:bias (hidden->bias later))))
    (check-true (close? causal
                        (scaled-dot-product-attention
                         square square v
                         #:mask (tril (ones 5 5 #:dtype 'bool)))))
    (check-true (close? (select causal 1 0) (select v 1 0))))

  (test-case "dropout is a no-op at zero and draws a seeded mask above it"
    (check-true (close? (scaled-dot-product-attention q k v #:dropout 0)
                        (scaled-dot-product-attention q k v)))
    (manual-seed! 1)
    (define dropped (scaled-dot-product-attention q k v #:dropout 0.5))
    (check-false (close? dropped (scaled-dot-product-attention q k v)))
    (manual-seed! 1)
    (check-true (close? dropped
                        (scaled-dot-product-attention q k v #:dropout 0.5))))

  (test-case "gradients reach the query, the key and the value"
    (define (leaves)
      (for/list ([t (in-list (list q k v))])
        (requires-grad! (detach t))))
    (define fused (leaves))
    (define by-hand (leaves))
    (backward! (sum (apply scaled-dot-product-attention fused)))
    (backward! (sum (apply attention-by-hand by-hand)))
    (for ([a (in-list fused)] [b (in-list by-hand)])
      (check-true (close? (grad a) (grad b) 1e-4))))

  (test-case "the contract states the shapes and refuses a mask with causal?"
    (define (mask-and-causal)
      (scaled-dot-product-attention q k v #:causal? #t
                                    #:mask (ones 3 5 #:dtype 'bool)))
    (check-exn (message-matching #rx"either #:mask or #:causal[?], not both")
               mask-and-causal)
    (check-exn (message-matching #rx"blaming: [(][^)]*attention-test[.]rkt")
               mask-and-causal)
    (check-not-exn
     (lambda () (scaled-dot-product-attention q k v #:causal? #f #:mask #f)))
    (check-exn (message-matching #rx"expected: attention-input")
               (lambda () (scaled-dot-product-attention (randn 4) k v)))
    (check-exn (message-matching #rx"query and key end in the same size, E")
               (lambda () (scaled-dot-product-attention (randn 2 3 5) k v)))
    (check-exn (message-matching #rx"key and value have the same length, S")
               (lambda () (scaled-dot-product-attention q k (randn 2 4 6))))
    (check-exn exn:fail:contract?
               (lambda () (scaled-dot-product-attention q k v #:dropout 1)))
    (check-exn exn:fail:contract?
               (lambda () (scaled-dot-product-attention q k v #:scale 'auto)))
    (check-exn exn:fail:contract?
               (lambda () (scaled-dot-product-attention q k v #:mask '(#t)))))

  (when (cuda-available?)
    (test-case "CUDA's fused kernels agree with the CPU"
      (define (on-cuda t) (to-device t 'cuda))
      (define bias (randn 3 5))
      (check-true
       (close? (scaled-dot-product-attention q k v #:mask bias)
               (to-device (scaled-dot-product-attention
                           (on-cuda q) (on-cuda k) (on-cuda v)
                           #:mask (on-cuda bias))
                          'cpu)
               1e-4))
      (define square (randn 2 5 4))
      (check-true
       (close? (scaled-dot-product-attention square square v #:causal? #t)
               (to-device (scaled-dot-product-attention
                           (on-cuda square) (on-cuda square) (on-cuda v)
                           #:causal? #t)
                          'cpu)
               1e-4))
      (check-true
       (close? (to-device (scaled-dot-product-attention
                           (on-cuda q) (on-cuda k) (on-cuda v)
                           #:mask (on-cuda (zeros 5 #:dtype 'bool)))
                          'cpu)
               (zeros 2 3 6)))
      (define half (to (on-cuda (randn 2 4 8 16)) 'bfloat16))
      (check-equal? (dtype (scaled-dot-product-attention half half half
                                                         #:causal? #t))
                    'bfloat16))))

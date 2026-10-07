#lang racket/base

(module+ test
  (require (only-in racket/list last make-list)
           rackunit
           (only-in "../generated.rkt" mean-dim sum-dim-intlist)
           "../main.rkt"
           "../nn.rkt")

  (define (sum-over-keys weights)
    (sum-dim-intlist weights '(-1) #f #f))

  (define (mean-over-heads weights)
    (mean-dim weights '(1) #f #f))

  (define (close? a b [eps 1e-5])
    (and (equal? (shape a) (shape b))
         (for/and ([x (in-list (tensor->list a))]
                   [y (in-list (tensor->list b))])
           (< (abs (- x y)) eps))))

  (define ((message-matching pattern) e)
    (and (exn:fail:contract? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define (param m name)
    (cdr (assoc name (named-parameters m))))

  (define (hidden->bias hidden)
    (masked-fill (zeros-like hidden #:dtype 'float32) hidden -inf.0))

  (define (heads-by-hand mha heads q k v #:bias [bias #f])
    (define projected-q ((child-ref mha "query") q))
    (define projected-k ((child-ref mha "key") k))
    (define projected-v ((child-ref mha "value") v))
    (define d (quotient (last (shape projected-q)) heads))
    (define (head t h) (narrow t -1 (* h d) d))
    ((child-ref mha "out")
     (cat (for/list ([h (in-range heads)])
            (define scores
              (* (@ (head projected-q h) (transpose (head projected-k h) -2 -1))
                 (/ 1 (sqrt d))))
            (@ (softmax (if bias (+ scores bias) scores) -1)
               (head projected-v h)))
          -1)))

  (define seeded
    (make-keyword-procedure
     (lambda (kws kw-args . args)
       (manual-seed! 0)
       (keyword-apply MultiheadAttention kws kw-args args))))

  (manual-seed! 1)
  (define x (randn 2 5 8))
  (define padded (eq (tensor '((0 0 0 0 0) (0 0 0 1 1))) 1))

  (test-case "four Linears named query, key, value and out"
    (define mha (MultiheadAttention 8 #:heads 2))
    (check-pred multihead-attention? mha)
    (check-pred layer? mha)
    (check-false (multihead-attention? (Linear 8 8)))
    (check-equal? (map car (named-children mha))
                  '("query" "key" "value" "out"))
    (for ([child (in-list (children mha))])
      (check-pred linear? child))
    (check-equal? (map car (named-parameters mha))
                  '("query.weight" "query.bias" "key.weight" "key.bias"
                    "value.weight" "value.bias" "out.weight" "out.bias"))
    (check-equal? (map shape (parameters mha))
                  '((8 8) (8) (8 8) (8) (8 8) (8) (8 8) (8)))
    (check-equal? (map car (named-parameters (MultiheadAttention 8 #:heads 2
                                                                 #:bias? #f)))
                  '("query.weight" "key.weight" "value.weight" "out.weight"))
    (define cross (MultiheadAttention 8 #:heads 4 #:key-dim 5 #:value-dim 3))
    (check-equal? (map shape (parameters cross))
                  '((8 8) (8) (8 5) (8) (8 3) (8) (8 8) (8))))

  (test-case "the draws follow nn.MultiheadAttention: out first, then xavier"
    (define mha (seeded 8 #:heads 2))
    (manual-seed! 0)
    (define out-proj (Linear 8 8))
    (define bound (* (sqrt 3.0) (sqrt (/ 2.0 32.0))))
    (define packed (uniform-init '(24 8) (- bound) bound))
    (check-equal? (tensor->list (param mha "out.weight"))
                  (tensor->list (param out-proj "weight")))
    (for ([name (in-list '("query" "key" "value"))] [i (in-naturals)])
      (check-equal? (tensor->list (param mha (string-append name ".weight")))
                    (tensor->list (narrow packed 0 (* 8 i) 8)))
      (check-equal? (tensor->list (param mha (string-append name ".bias")))
                    (make-list 8 0.0)))
    (check-equal? (tensor->list (param mha "out.bias")) (make-list 8 0.0))
    (check-equal? (tensor->list (randn 3))
                  (begin (seeded 8 #:heads 2) (tensor->list (randn 3)))
                  "the stream ends where it would after nn.MultiheadAttention")
    (define cross (seeded 8 #:heads 2 #:key-dim 5 #:value-dim 6))
    (manual-seed! 0)
    (Linear 8 8)
    (define (xavier fan-out fan-in)
      (define std (sqrt (/ 2.0 (exact->inexact (+ fan-in fan-out)))))
      (define bound (* (sqrt 3.0) std))
      (uniform-init (list fan-out fan-in) (- bound) bound))
    (for ([name (in-list '("query" "key" "value"))]
          [width (in-list '(8 5 6))])
      (check-equal? (tensor->list (param cross (string-append name ".weight")))
                    (tensor->list (xavier 8 width)))))

  (test-case "each head attends on its own slice of the projections"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (check-true (close? (mha x x x) (heads-by-hand mha 2 x x x)))
    (define four (seeded 8 #:heads 4 #:batch-first? #t))
    (check-true (close? (four x x x) (heads-by-hand four 4 x x x)))
    (check-false (close? (four x x x) (mha x x x))))

  (test-case "sequence-first by default, batch-first on request, or unbatched"
    (define sequence-first (seeded 8 #:heads 2))
    (define batch-first (seeded 8 #:heads 2 #:batch-first? #t))
    (define xt (transpose x 0 1))
    (check-equal? (shape (sequence-first xt xt xt)) '(5 2 8))
    (check-true (close? (transpose (sequence-first xt xt xt) 0 1)
                        (batch-first x x x)))
    (define one (select x 0 0))
    (check-equal? (shape (batch-first one one one)) '(5 8))
    (check-true (close? (batch-first one one one)
                        (select (batch-first x x x) 0 0)))
    (check-true (close? (sequence-first one one one)
                        (batch-first one one one)))
    (define-values (_out weights)
      (sequence-first one one one #:need-weights? #t))
    (check-equal? (shape weights) '(5 5))
    (define-values (_out2 per-head)
      (sequence-first one one one #:need-weights? #t
                      #:average-attn-weights? #f))
    (check-equal? (shape per-head) '(2 5 5)))

  (test-case "the weights are a distribution over the keys, per query"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (define-values (out weights) (mha x x x #:need-weights? #t))
    (check-equal? (shape weights) '(2 5 5))
    (check-true (close? (sum-over-keys weights) (ones 2 5)))
    (check-true (close? out (mha x x x)) "the fused path agrees")
    (define-values (_out per-head)
      (mha x x x #:need-weights? #t #:average-attn-weights? #f))
    (check-equal? (shape per-head) '(2 2 5 5))
    (check-true (close? (sum-over-keys per-head) (ones 2 2 5)))
    (check-true (close? (mean-over-heads per-head) weights)))

  (test-case "a key-padding mask hides the keys where it is #t"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (define-values (out weights)
      (mha x x x #:key-padding-mask padded #:need-weights? #t))
    (check-equal? (tensor->list (ref weights 1 : (: 3 5)))
                  (make-list 10 0.0))
    (check-true (close? (sum-over-keys weights) (ones 2 5)))
    (define bias (reshape (hidden->bias padded) 2 1 5))
    (check-true (close? out (heads-by-hand mha 2 x x x #:bias bias)))
    (check-true (close? (mha x x x #:key-padding-mask padded) out)
                "the fused path agrees")
    (check-true (close? (mha x x x #:key-padding-mask (hidden->bias padded))
                        out)
                "a float mask is added to the scores")
    (check-true (close? (select out 0 0) (select (mha x x x) 0 0))
                "an unpadded row is unchanged"))

  (test-case "the opposite sense to scaled-dot-product-attention's mask"
    (define mha (seeded 8 #:heads 1 #:batch-first? #t))
    (define later (eq (tril (ones 5 5)) 0))
    (define-values (_out weights)
      (mha x x x #:attn-mask later #:need-weights? #t))
    (check-equal? (tensor->list (ref weights 0 0 (: 1 5)))
                  (make-list 4 0.0)
                  "#t hides the key")
    (define q ((child-ref mha "query") x))
    (define k ((child-ref mha "key") x))
    (define v ((child-ref mha "value") x))
    (check-true (close? (mha x x x #:attn-mask later)
                        ((child-ref mha "out")
                         (scaled-dot-product-attention
                          q k v #:mask (eq later 0))))))

  (test-case "attn-mask is [L, S] for every head or [N·heads, L, S] each"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (define bias (randn 5 5))
    (check-true (close? (mha x x x #:attn-mask bias)
                        (heads-by-hand mha 2 x x x #:bias bias)))
    (define per-head (eq (tensor (for/list ([i (in-range 4)])
                                   (for/list ([q (in-range 5)])
                                     (for/list ([s (in-range 5)])
                                       (if (= s i) 1 0)))))
                         1))
    (define-values (_out weights)
      (mha x x x #:attn-mask per-head #:need-weights? #t
           #:average-attn-weights? #f))
    (for* ([n (in-range 2)] [h (in-range 2)])
      (define hidden (+ (* 2 n) h))
      (check-equal? (tensor->list (ref weights n h : hidden))
                    (make-list 5 0.0)
                    (format "batch ~a head ~a hides key ~a" n h hidden))))

  (test-case "masks combine: a key either mask hides stays hidden"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (define first-key (eq (tensor '((1 0 0 0 0) (0 0 0 0 0) (0 0 0 0 0)
                                    (0 0 0 0 0) (0 0 0 0 0)))
                          1))
    (define-values (out weights)
      (mha x x x #:key-padding-mask padded #:attn-mask first-key
           #:need-weights? #t))
    (check-equal? (ref weights 1 0 0) 0.0)
    (check-equal? (ref weights 1 1 4) 0.0)
    (define bias (+ (reshape (hidden->bias padded) 2 1 5)
                    (hidden->bias first-key)))
    (check-true (close? out (heads-by-hand mha 2 x x x #:bias bias)))
    (check-true (close? (mha x x x #:key-padding-mask padded
                             #:attn-mask first-key)
                        out)))

  (test-case "#:causal? hides every later key, fused or with the weights"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (define later (eq (tril (ones 5 5)) 0))
    (define-values (out weights) (mha x x x #:causal? #t #:need-weights? #t))
    (check-equal? (tensor->list (triu (select weights 0 0) 1))
                  (make-list 25 0.0))
    (check-true (close? out (mha x x x #:causal? #t)))
    (check-true (close? out (mha x x x #:attn-mask later)))
    (check-true (close? out (heads-by-hand mha 2 x x x
                                           #:bias (hidden->bias later))))
    (define both (mha x x x #:causal? #t #:key-padding-mask padded))
    (check-true (close? both
                        (heads-by-hand mha 2 x x x
                                       #:bias (+ (hidden->bias later)
                                                 (reshape (hidden->bias padded)
                                                          2 1 5)))))
    (check-true (close? (mha x x x #:causal? #t #:attn-mask later) out)
                "a causal mask passed again changes nothing"))

  (test-case "cross-attention: keys and values of their own length and width"
    (define mha (seeded 8 #:heads 2 #:key-dim 5 #:value-dim 3
                        #:batch-first? #t))
    (define memory-k (randn 2 7 5))
    (define memory-v (randn 2 7 3))
    (define-values (out weights)
      (mha x memory-k memory-v #:need-weights? #t))
    (check-equal? (shape out) '(2 5 8))
    (check-equal? (shape weights) '(2 5 7))
    (check-true (close? out (heads-by-hand mha 2 x memory-k memory-v)))
    (define gaps (eq (tensor '((0 0 0 0 0 0 0) (0 0 0 0 1 1 1))) 1))
    (check-true (close? (mha x memory-k memory-v #:key-padding-mask gaps)
                        (heads-by-hand mha 2 x memory-k memory-v
                                       #:bias (reshape (hidden->bias gaps)
                                                       2 1 7))))
    (define memory (randn 2 7 8))
    (define same-width (seeded 8 #:heads 2 #:batch-first? #t))
    (check-true (close? (same-width x memory memory)
                        (heads-by-hand same-width 2 x memory memory))))

  (test-case "dropout on the weights in training mode only"
    (define plain (seeded 8 #:heads 2 #:batch-first? #t))
    (define dropping (seeded 8 #:heads 2 #:batch-first? #t #:dropout 0.5))
    (define reference (plain x x x))
    (check-false (close? (dropping x x x) reference))
    (define-values (_out weights) (dropping x x x #:need-weights? #t))
    (check-false (close? (sum-over-keys weights) (ones 2 5)))
    (eval! dropping)
    (check-true (close? (dropping x x x) reference))
    (define-values (_eval-out eval-weights)
      (dropping x x x #:need-weights? #t))
    (check-true (close? (sum-over-keys eval-weights) (ones 2 5)))
    (train! dropping)
    (manual-seed! 7)
    (define first (dropping x x x))
    (manual-seed! 7)
    (check-true (close? (dropping x x x) first) "seeded dropout replays"))

  (test-case "gradients reach every projection and every input"
    (define (grads need-weights?)
      (define mha (seeded 8 #:heads 2 #:batch-first? #t))
      (define inputs
        (map requires-grad! (list (randn 2 5 8) (randn 2 7 8))))
      (define q (car inputs))
      (define memory (cadr inputs))
      (define out
        (if need-weights?
            (let-values ([(out _w) (mha q memory memory #:need-weights? #t)])
              out)
            (mha q memory memory)))
      (backward! (sum (* out out)))
      (append (map grad inputs) (map grad (parameters mha))))
    (define fused (grads #f))
    (define explicit (grads #t))
    (check-equal? (length fused) 10)
    (for ([a (in-list fused)] [b (in-list explicit)])
      (check-true (close? a b 1e-4)))
    (for ([g (in-list fused)])
      (check-true (> (item (sum (abs g))) 0.0))))

  (test-case "to moves the four projections, and the masks follow the input"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (check-eq? (to mha 'float64) mha)
    (check-equal? (map dtype (parameters mha)) (make-list 8 'float64))
    (define wide (to x 'float64))
    (define-values (out weights)
      (mha wide wide wide #:key-padding-mask padded #:causal? #t
           #:attn-mask (randn 5 5) #:need-weights? #t))
    (check-equal? (dtype out) 'float64)
    (check-equal? (dtype weights) 'float64))

  (test-case "the constructor's contract blames its caller"
    (define blames-this-test
      (message-matching #rx"blaming: [(][^)]*multihead-attention-test[.]rkt"))
    (check-exn (message-matching #rx"#:heads divides the embedding width")
               (lambda () (MultiheadAttention 8 #:heads 3)))
    (check-exn blames-this-test (lambda () (MultiheadAttention 8 #:heads 3)))
    (check-exn #rx"^MultiheadAttention: contract violation"
               (lambda () (MultiheadAttention 8 #:heads 2 #:dropout 1)))
    (check-exn exn:fail:contract?
               (lambda () (MultiheadAttention 8 #:heads 2 #:key-dim 0)))
    (check-exn exn:fail:contract? (lambda () (MultiheadAttention 8)))
    (check-exn exn:fail:contract?
               (lambda () (MultiheadAttention 0 #:heads 1))))

  (test-case "an application the layer cannot take is the caller's violation"
    (define mha (seeded 8 #:heads 2 #:batch-first? #t))
    (define blames-caller (message-matching #rx"blaming: caller"))
    (define (refuses pattern thunk)
      (check-exn (message-matching pattern) thunk)
      (check-exn (message-matching #rx"^MultiheadAttention: contract violation")
                 thunk)
      (check-exn blames-caller thunk))
    (refuses #rx"query is 7 wide, not 8"
             (lambda () (mha (randn 2 5 7) x x)))
    (refuses #rx"value is 6 wide, not 8"
             (lambda () (mha x x (randn 2 5 6))))
    (refuses #rx"key is 4 wide, not 8"
             (lambda () (mha x (randn 2 5 4) x)))
    (refuses #rx"expected: attention-sequence"
             (lambda () (mha (randn 1 2 5 8) x x)))
    (refuses #rx"all batched [(]rank 3[)] or all unbatched"
             (lambda () (mha (select x 0 0) x x)))
    (refuses (regexp (string-append "key and value differ in batch or length:"
                                    " [(]2 5[)] and [(]2 4[)]"))
             (lambda () (mha x x (randn 2 4 8))))
    (refuses #rx"query has batch 3 and key 2"
             (lambda () (mha (randn 3 5 8) x x)))
    (refuses #rx"key-padding-mask has shape [(]2 4[)], not [(]2 5[)]"
             (lambda ()
               (mha x x x #:key-padding-mask (zeros 2 4 #:dtype 'bool))))
    (refuses (regexp (string-append "attn-mask has shape [(]5 4[)],"
                                    " not [(]5 5[)] or [(]4 5 5[)]"))
             (lambda () (mha x x x #:attn-mask (zeros 5 4))))
    (refuses #rx"expected: [(]or/c #f attention-mask[)]"
             (lambda () (mha x x x #:attn-mask (zeros 5 5 #:dtype 'int64))))
    (refuses #rx"expected: boolean[?]"
             (lambda () (mha x x x #:causal? 'yes)))
    (check-exn #rx"MultiheadAttention: arity mismatch"
               (lambda () (mha x x)))
    (check-exn #rx"procedure: MultiheadAttention.*given keyword: #:mask"
               (lambda () (mha x x x #:mask padded))))

  (when (cuda-available?)
    (test-case "on CUDA both paths agree with the CPU, masks included"
      (define mha (seeded 8 #:heads 2 #:batch-first? #t))
      (define (each-path x padded bias)
        (list (mha x x x #:key-padding-mask padded #:attn-mask bias)
              (mha x x x #:causal? #t)
              (let-values ([(out weights)
                            (mha x x x #:causal? #t #:key-padding-mask padded
                                 #:need-weights? #t)])
                (cat (list (flatten out) (flatten weights))))))
      (define bias (randn 5 5))
      (define on-cpu (each-path x padded bias))
      (to mha 'cuda)
      (define (on-cuda t) (to-device t 'cuda))
      (for ([a (in-list on-cpu)]
            [b (in-list (each-path (on-cuda x) (on-cuda padded)
                                   (on-cuda bias)))])
        (check-true (close? a (to-device b 'cpu) 1e-4))))))

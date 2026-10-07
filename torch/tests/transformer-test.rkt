#lang racket/base

(module+ test
  (require (only-in racket/list remove-duplicates take-right)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "../vision/diffusion.rkt" sinusoidal-embedding))

  (define (close? a b [eps 1e-5])
    (and (equal? (shape a) (shape b))
         (for/and ([x (in-flattened-tensor a)] [y (in-flattened-tensor b)])
           (< (abs (- x y)) eps))))

  (define ((message-matching pattern) e)
    (and (exn:fail:contract? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define (named m)
    (map car (named-parameters m)))

  (define (child m . path)
    (for/fold ([m m]) ([name (in-list path)])
      (child-ref m name)))

  (define ((seeded constructor)
           #:dropout [p 0.0]
           #:activation [activation 'relu]
           #:norm-first? [norm-first? #f]
           #:batch-first? [batch-first? #f]
           #:bias? [bias? #t])
    (manual-seed! 0)
    (constructor 8 #:heads 2 #:ffn-width 16 #:dropout p
                 #:activation activation #:norm-first? norm-first?
                 #:batch-first? batch-first? #:bias? bias?))

  (define encoder-layer (seeded TransformerEncoderLayer))
  (define decoder-layer (seeded TransformerDecoderLayer))

  (define (layers-of stack)
    (for/list ([l (in-layers (child-ref stack "layers"))]) l))

  (manual-seed! 1)
  (define x (randn 5 2 8))
  (define memory (randn 7 2 8))
  (define padded (eq (tensor '((0 0 0 0 0) (0 0 0 1 1))) 1))

  (define (attend-by-hand attention q kv
                          #:mask [mask #f] #:padding [padding #f]
                          #:causal? [causal? #f])
    (attention q kv kv #:attn-mask mask #:key-padding-mask padding
               #:causal? causal?))

  (define (encoder-by-hand layer src activation pre-norm?
                           #:mask [mask #f] #:padding [padding #f]
                           #:causal? [causal? #f])
    (define (attend x)
      (attend-by-hand (child layer "self-attn") x x
                      #:mask mask #:padding padding #:causal? causal?))
    (define (feed x)
      ((child layer "linear2") (activation ((child layer "linear1") x))))
    (define norm1 (child layer "norm1"))
    (define norm2 (child layer "norm2"))
    (cond
      [pre-norm?
       (define h (+ src (attend (norm1 src))))
       (+ h (feed (norm2 h)))]
      [else
       (define h (norm1 (+ src (attend src))))
       (norm2 (+ h (feed h)))]))

  (test-case "an encoder layer has PyTorch's children, in its order"
    (define layer (encoder-layer))
    (check-pred transformer-encoder-layer? layer)
    (check-pred layer? layer)
    (check-false (transformer-encoder-layer? (Linear 2 2)))
    (check-equal? (map car (named-children layer))
                  '("self-attn" "linear1" "dropout" "linear2" "norm1" "norm2"
                    "dropout1" "dropout2"))
    (check-equal? (named layer)
                  '("self-attn.query.weight" "self-attn.query.bias"
                    "self-attn.key.weight" "self-attn.key.bias"
                    "self-attn.value.weight" "self-attn.value.bias"
                    "self-attn.out.weight" "self-attn.out.bias"
                    "linear1.weight" "linear1.bias"
                    "linear2.weight" "linear2.bias"
                    "norm1.weight" "norm1.bias" "norm2.weight" "norm2.bias"))
    (check-equal? (map shape (parameters layer))
                  '((8 8) (8) (8 8) (8) (8 8) (8) (8 8) (8)
                    (16 8) (16) (8 16) (8) (8) (8) (8) (8)))
    (check-equal? (named (encoder-layer #:bias? #f))
                  '("self-attn.query.weight" "self-attn.key.weight"
                    "self-attn.value.weight" "self-attn.out.weight"
                    "linear1.weight" "linear2.weight"
                    "norm1.weight" "norm2.weight"))
    (check-equal? (map shape
                       (parameters (child (TransformerEncoderLayer 8 #:heads 2)
                                          "linear1")))
                  '((2048 8) (2048))
                  "PyTorch's 2048-wide feed-forward by default"))

  (test-case "a decoder layer adds cross-attention and a third norm"
    (define layer (decoder-layer))
    (check-pred transformer-decoder-layer? layer)
    (check-false (transformer-decoder-layer? (encoder-layer)))
    (check-equal? (map car (named-children layer))
                  '("self-attn" "multihead-attn" "linear1" "dropout" "linear2"
                    "norm1" "norm2" "norm3" "dropout1" "dropout2" "dropout3"))
    (check-equal? (length (parameters layer)) 26)
    (check-equal? (length (parameters (decoder-layer #:bias? #f))) 13)
    (check-equal? (map shape
                       (parameters (child (TransformerDecoderLayer 8 #:heads 2)
                                          "linear2")))
                  '((8 2048) (8))))

  (test-case "the draws follow PyTorch: attention, then linear1 and linear2"
    (define layer (encoder-layer))
    (define after (tensor->list (randn 3)))
    (manual-seed! 0)
    (define mha (MultiheadAttention 8 #:heads 2 #:dropout 0.0))
    (define l1 (Linear 8 16))
    (define l2 (Linear 16 8))
    (check-equal? (tensor->list (randn 3)) after
                  "the stream ends where it would after PyTorch's layer")
    (for ([a (in-list (parameters layer))]
          [b (in-list (append (parameters mha) (parameters l1)
                              (parameters l2)))])
      (check-equal? (tensor->list a) (tensor->list b)))
    (define decoder (decoder-layer))
    (manual-seed! 0)
    (define self (MultiheadAttention 8 #:heads 2))
    (define cross (MultiheadAttention 8 #:heads 2))
    (for ([a (in-list (parameters decoder))]
          [b (in-list (append (parameters self) (parameters cross)
                              (parameters (Linear 8 16))
                              (parameters (Linear 16 8))))])
      (check-equal? (tensor->list a) (tensor->list b))))

  (test-case "post-norm by default, pre-norm with #:norm-first?"
    (define post (encoder-layer))
    (define pre (encoder-layer #:norm-first? #t))
    (check-true (close? (post x) (encoder-by-hand post x relu #f)))
    (check-true (close? (pre x) (encoder-by-hand pre x relu #t)))
    (check-false (close? (pre x) (post x))))

  (test-case "relu, gelu, tanh gelu or any procedure in the feed-forward"
    (define (tanh-gelu t) (gelu t #:approximate 'tanh))
    (for ([activation (in-list (list 'gelu 'gelu-tanh tanh-gelu sigmoid))]
          [by-hand (in-list (list gelu tanh-gelu tanh-gelu sigmoid))])
      (define layer (encoder-layer #:activation activation #:norm-first? #t))
      (check-true (close? (layer x) (encoder-by-hand layer x by-hand #t))
                  (format "~a" activation)))
    (check-false (close? ((encoder-layer #:activation 'gelu) x)
                         ((encoder-layer #:activation 'gelu-tanh) x)
                         1e-7)))

  (test-case "sequence-first by default, batch-first on request, or unbatched"
    (define seq (encoder-layer))
    (define batch (encoder-layer #:batch-first? #t))
    (define xt (transpose x 0 1))
    (check-equal? (shape (seq x)) '(5 2 8))
    (check-equal? (shape (batch xt)) '(2 5 8))
    (check-true (close? (transpose (seq x #:key-padding-mask padded) 0 1)
                        (batch xt #:key-padding-mask padded)))
    (define one (select x 1 0))
    (check-equal? (shape (seq one)) '(5 8))
    (check-true (close? (seq one) (select (seq x) 1 0)))
    (define dseq (decoder-layer))
    (define dbatch (decoder-layer #:batch-first? #t))
    (define tgt (randn 4 2 8))
    (check-equal? (shape (dseq tgt memory)) '(4 2 8))
    (check-true (close? (transpose (dseq tgt memory #:tgt-causal? #t) 0 1)
                        (dbatch (transpose tgt 0 1) (transpose memory 0 1)
                                #:tgt-causal? #t)))
    (check-true (close? (dseq (select tgt 1 0) (select memory 1 0))
                        (select (dseq tgt memory) 1 0))))

  (test-case "masks reach the self-attention, in MultiheadAttention's sense"
    (define layer (encoder-layer))
    (check-true (close? (layer x #:key-padding-mask padded)
                        (encoder-by-hand layer x relu #f #:padding padded)))
    (define hide-later (causal-mask 5))
    (check-true (close? (layer x #:causal? #t)
                        (layer x #:mask hide-later)))
    (check-true (close? (layer x #:mask (causal-mask 5 #:dtype 'float32))
                        (layer x #:causal? #t)))
    (define bias (randn 5 5))
    (check-true (close? (layer x #:mask bias #:key-padding-mask padded)
                        (encoder-by-hand layer x relu #f #:mask bias
                                         #:padding padded)))
    (define changed (cat (list (narrow x 0 0 3) (randn 2 2 8)) 0))
    (check-true (close? (narrow (layer x #:causal? #t) 0 0 3)
                        (narrow (layer changed #:causal? #t) 0 0 3))
                "a causal layer's early positions never read the later ones")
    (define repadded
      (cat (list (narrow x 0 0 3)
                 (stack (list (select (narrow x 0 3 2) 1 0) (randn 2 8)) 1))
           0))
    (check-true (close? (narrow (layer x #:key-padding-mask padded) 0 0 3)
                        (narrow (layer repadded #:key-padding-mask padded)
                                0 0 3))
                "what sits under the padding changes no other position"))

  (test-case "a decoder hides its own future and the memory's padding"
    (define layer (decoder-layer #:norm-first? #t))
    (define tgt (randn 4 2 8))
    (define changed (cat (list (narrow tgt 0 0 2) (randn 2 2 8)) 0))
    (check-true (close? (narrow (layer tgt memory #:tgt-causal? #t) 0 0 2)
                        (narrow (layer changed memory #:tgt-causal? #t) 0 0 2)))
    (check-false (close? (narrow (layer tgt memory) 0 0 2)
                         (narrow (layer changed memory) 0 0 2)))
    (check-true (close? (layer tgt memory #:tgt-causal? #t)
                        (layer tgt memory #:tgt-mask (causal-mask 4))))
    (define gaps (eq (tensor '((0 0 0 0 0 0 0) (0 0 0 0 0 1 1))) 1))
    (define other (cat (list (narrow memory 0 0 5) (randn 2 2 8)) 0))
    (define masked (layer tgt memory #:memory-key-padding-mask gaps))
    (check-true (close? (select masked 1 1)
                        (select (layer tgt other #:memory-key-padding-mask gaps)
                                1 1)))
    (define first-three (eq (tril (ones 4 7) 2) 0))
    (check-true (close? (layer tgt memory #:memory-mask first-three
                               #:tgt-key-padding-mask (eq (tensor '((0 0 0 0)
                                                                    (0 0 0 1)))
                                                          1))
                        (layer tgt memory #:memory-mask first-three
                               #:tgt-key-padding-mask
                               (masked-fill (zeros 2 4)
                                            (eq (tensor '((0 0 0 0) (0 0 0 1)))
                                                1)
                                            -inf.0))))
    (check-false (close? (layer tgt memory #:memory-causal? #t)
                         (layer tgt memory))))

  (test-case "dropout in training mode only"
    (define plain (encoder-layer))
    (define dropping (encoder-layer #:dropout 0.5))
    (check-false (close? (dropping x) (plain x)))
    (eval! dropping)
    (check-true (close? (dropping x) (plain x)))
    (check-false (layer-training? (child dropping "self-attn")))
    (check-false (layer-training? (child dropping "dropout2")))
    (train! dropping)
    (manual-seed! 3)
    (define once (dropping x))
    (manual-seed! 3)
    (check-true (close? (dropping x) once) "seeded dropout replays")
    (define decoder (decoder-layer #:dropout 0.5))
    (define tgt (randn 4 2 8))
    (check-false (close? (decoder tgt memory) (decoder tgt memory)))
    (eval! decoder)
    (check-true (close? (decoder tgt memory) (decoder tgt memory))))

  (test-case "a stack copies one layer, as nn.TransformerEncoder does"
    (manual-seed! 0)
    (define stack
      (TransformerEncoder 8 #:heads 2 #:ffn-width 16 #:layers 3 #:norm #t))
    (define after (tensor->list (randn 3)))
    (check-pred transformer-encoder? stack)
    (check-equal? (map car (named-children stack)) '("layers" "norm"))
    (check-equal? (length (parameters stack)) (+ (* 3 16) 2))
    (check-equal? (car (named stack)) "layers.0.self-attn.query.weight")
    (check-equal? (take-right (named stack) 4)
                  '("layers.2.norm2.weight" "layers.2.norm2.bias"
                    "norm.weight" "norm.bias"))
    (define layers (layers-of stack))
    (check-equal? (length layers) 3)
    (for ([l (in-list (cdr layers))])
      (for ([a (in-list (parameters l))]
            [b (in-list (parameters (car layers)))])
        (check-false (eq? a b) "copies, not one layer shared")
        (check-equal? (tensor->list a) (tensor->list b))))
    (manual-seed! 0)
    (define one (TransformerEncoderLayer 8 #:heads 2 #:ffn-width 16))
    (check-equal? (tensor->list (randn 3)) after
                  "the copies draw nothing")
    (for ([a (in-list (parameters one))]
          [b (in-list (parameters (car layers)))])
      (check-equal? (tensor->list a) (tensor->list b)))
    (check-pred layer-norm? (child stack "norm"))
    (eval! stack)
    (check-true (close? (stack x #:key-padding-mask padded #:causal? #t)
                        ((child stack "norm")
                         (for/fold ([h x]) ([l (in-list layers)])
                           (l h #:key-padding-mask padded #:causal? #t)))))
    (define bare (TransformerEncoder 8 #:heads 2 #:ffn-width 16 #:layers 1))
    (check-equal? (map car (named-children bare)) '("layers"))
    (eval! bare)
    (check-true (close? (bare x) ((car (layers-of bare)) x))))

  (test-case "a stack's defaults are its layer's"
    (manual-seed! 0)
    (define encoder (TransformerEncoder 8 #:heads 2 #:layers 2))
    (manual-seed! 0)
    (define one (TransformerEncoderLayer 8 #:heads 2))
    (check-equal? (map shape (parameters encoder))
                  (append (map shape (parameters one))
                          (map shape (parameters one))))
    (check-false (close? (encoder x) (encoder x)) "dropout 0.1 in training")
    (eval! encoder)
    (eval! one)
    (check-true (close? (encoder x) (one (one x))))
    (manual-seed! 0)
    (define decoder (TransformerDecoder 8 #:heads 2 #:layers 2))
    (manual-seed! 0)
    (define one-decoder (TransformerDecoderLayer 8 #:heads 2))
    (check-equal? (map car (named-children decoder)) '("layers"))
    (eval! decoder)
    (eval! one-decoder)
    (define tgt (randn 4 2 8))
    (check-true (close? (decoder tgt memory)
                        (one-decoder (one-decoder tgt memory) memory))))

  (test-case "a stack hands its keywords to every layer and to its norm"
    (manual-seed! 0)
    (define stack
      (TransformerDecoder 8 #:heads 2 #:ffn-width 16 #:layers 2 #:dropout 0.0
                          #:norm-first? #t #:activation 'gelu-tanh
                          #:batch-first? #t #:bias? #f #:layer-norm-eps 1e-3
                          #:norm #t))
    (manual-seed! 0)
    (define one
      (TransformerDecoderLayer 8 #:heads 2 #:ffn-width 16 #:dropout 0.0
                               #:norm-first? #t #:activation 'gelu-tanh
                               #:batch-first? #t #:bias? #f
                               #:layer-norm-eps 1e-3))
    (check-pred transformer-decoder? stack)
    (check-equal? (length (parameters stack)) (+ (* 2 13) 1))
    (check-equal? (take-right (named stack) 2)
                  '("layers.1.norm3.weight" "norm.weight"))
    (define tgt (randn 2 4 8))
    (define mem (randn 2 7 8))
    (define gaps (eq (tensor '((0 0 0 0 0 0 0) (0 0 0 0 0 1 1))) 1))
    (define final (LayerNorm 8 #:eps 1e-3 #:bias? #f))
    (check-true
     (close? (stack tgt mem #:tgt-causal? #t #:memory-key-padding-mask gaps)
             (final (for/fold ([h tgt]) ([_ (in-range 2)])
                      (one h mem #:tgt-causal? #t
                           #:memory-key-padding-mask gaps))))))

  (test-case "a layer stacks as copies of it, the layer itself left out"
    (manual-seed! 0)
    (define width
      (TransformerEncoder 8 #:heads 2 #:ffn-width 16 #:layers 2 #:norm #t))
    (manual-seed! 0)
    (define prototype (TransformerEncoderLayer 8 #:heads 2 #:ffn-width 16))
    (define after (tensor->list (randn 3)))
    (manual-seed! 0)
    (TransformerEncoderLayer 8 #:heads 2 #:ffn-width 16)
    (check-equal? (tensor->list (randn 3)) after)
    (define stacked
      (TransformerEncoder prototype #:layers 2 #:norm (LayerNorm 8)))
    (check-pred transformer-encoder? width)
    (check-pred transformer-encoder? stacked)
    (check-false (transformer-decoder? stacked))
    (check-equal? (named stacked) (named width))
    (for ([a (in-list (parameters stacked))]
          [b (in-list (parameters width))])
      (check-equal? (tensor->list a) (tensor->list b)))
    (for ([l (in-list (layers-of stacked))])
      (check-false (eq? l prototype))
      (for ([a (in-list (parameters l))]
            [b (in-list (parameters prototype))])
        (check-false (eq? a b))))
    (with-no-grad
      (for ([p (in-list (parameters prototype))]) (zero! p)))
    (check-equal? (tensor->list (car (parameters stacked)))
                  (tensor->list (car (parameters width)))
                  "the stack does not share the given layer's tensors")
    (define custom
      (TransformerEncoder (encoder-layer) #:layers 1 #:norm (Linear 8 8)))
    (check-equal? (take-right (named custom) 2) '("norm.weight" "norm.bias"))
    (eval! custom)
    (check-true (close? (custom x)
                        ((child custom "norm") ((car (layers-of custom)) x)))))

  (test-case "a procedure stacks one fresh layer per call"
    (define (make) (TransformerDecoderLayer 8 #:heads 2 #:ffn-width 16))
    (manual-seed! 0)
    (define stack (TransformerDecoder make #:layers 2))
    (define after (tensor->list (randn 3)))
    (manual-seed! 0)
    (define first-layer (make))
    (define second-layer (make))
    (check-equal? (tensor->list (randn 3)) after)
    (define layers (layers-of stack))
    (for ([a (in-list (parameters (cadr layers)))]
          [b (in-list (parameters second-layer))])
      (check-equal? (tensor->list a) (tensor->list b)))
    (check-false (equal? (tensor->list (car (parameters (car layers))))
                         (tensor->list (car (parameters (cadr layers))))))
    (check-pred transformer-decoder? stack)
    (check-equal? (length (parameters stack)) 52)
    (define tgt (randn 4 2 8))
    (eval! stack)
    (check-true (close? (stack tgt memory #:tgt-causal? #t)
                        (for/fold ([h tgt]) ([l (in-list layers)])
                          (l h memory #:tgt-causal? #t))))
    (define copied (TransformerDecoder (make) #:layers 2))
    (check-equal? (tensor->list (car (parameters (car (layers-of copied)))))
                  (tensor->list (car (parameters (cadr (layers-of copied)))))))

  (test-case "gradients reach every parameter of a stack and its inputs"
    (manual-seed! 0)
    (define stack
      (TransformerDecoder 8 #:heads 2 #:ffn-width 16 #:layers 2
                          #:norm-first? #t #:activation 'gelu-tanh
                          #:norm #t))
    (define tgt (requires-grad! (randn 4 2 8)))
    (define mem (requires-grad! (randn 7 2 8)))
    (define out (stack tgt mem #:tgt-causal? #t))
    (backward! (sum (* out out)))
    (for ([g (in-list (append (list (grad tgt) (grad mem))
                              (map grad (parameters stack))))])
      (check-true (> (item (sum (abs g))) 0.0))))

  (test-case "to moves a layer, and the masks follow the input"
    (define layer (encoder-layer #:norm-first? #t))
    (check-eq? (to layer 'float64) layer)
    (check-equal? (remove-duplicates (map dtype (parameters layer)))
                  '(float64))
    (check-equal? (dtype (layer (to x 'float64)
                                #:key-padding-mask padded
                                #:causal? #t
                                #:mask (randn 5 5)))
                  'float64))

  (test-case "LayerNorm without a bias, as bias=False builds it"
    (define norm (LayerNorm 8 #:bias? #f))
    (check-equal? (named norm) '("weight"))
    (check-true (close? (norm x) (layer-norm x 8))))

  (test-case "sinusoidal positions: interleaved by default, or in halves"
    (define p (sinusoidal-positions 6 8))
    (check-equal? (shape p) '(6 8))
    (for* ([pos (in-range 6)] [i (in-range 4)])
      (define angle (* pos (exp (* i (- (/ (log 10000.0) 4))))))
      (check-= (ref p pos (* 2 i)) (sin angle) 1e-5)
      (check-= (ref p pos (add1 (* 2 i))) (cos angle) 1e-5))
    (define halves (sinusoidal-positions 6 8 #:layout 'halves))
    (check-true (close? (narrow halves 1 0 4) (ref p : (: 0 8 2))))
    (check-true (close? (narrow halves 1 4 4) (ref p : (: 1 8 2))))
    (define steps (tensor '(0 3 999) #:dtype 'int64))
    (check-equal? (tensor->list
                   (sinusoidal-positions steps 16 #:layout 'halves))
                  (tensor->list (sinusoidal-embedding steps 16))
                  "the halves layout is the diffusion embedding, bit for bit")
    (check-true (close? (sinusoidal-positions (tensor '(2 4)) 8)
                        (index-select p 0 (tensor '(2 4)))))
    (check-equal? (shape (sinusoidal-positions 0 4)) '(0 4))
    (check-equal? (device (sinusoidal-positions 2 4 #:device 'cpu))
                  (cpu-device)))

  (test-case "the causal mask is #t above the diagonal, or -inf as a float"
    (check-equal? (tensor->list (causal-mask 3))
                  '(0.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0))
    (check-equal? (dtype (causal-mask 3)) 'bool)
    (check-equal? (tensor->list (causal-mask 2 #:dtype 'float64))
                  '(0.0 -inf.0 0.0 0.0))
    (check-equal? (dtype (causal-mask 2 #:dtype 'float64)) 'float64)
    (check-equal? (shape (causal-mask 0)) '(0 0))
    (define mha (MultiheadAttention 8 #:heads 2))
    (check-true (close? (mha x x x #:attn-mask (causal-mask 5))
                        (mha x x x #:causal? #t))))

  (test-case "the constructors' contracts blame their caller"
    (define blames-this-test
      (message-matching #rx"blaming: [(][^)]*transformer-test[.]rkt"))
    (check-exn (message-matching #rx"#:heads divides the model width")
               (lambda () (TransformerEncoderLayer 8 #:heads 3)))
    (check-exn blames-this-test
               (lambda () (TransformerDecoderLayer 8 #:heads 3)))
    (check-exn blames-this-test
               (lambda () (TransformerEncoderLayer 8 #:heads 2
                                                   #:activation 'tanh)))
    (check-exn #rx"^TransformerEncoderLayer: contract violation"
               (lambda () (TransformerEncoderLayer 8 #:heads 2 #:dropout 1)))
    (check-exn exn:fail:contract?
               (lambda () (TransformerEncoderLayer 8 #:heads 2
                                                   #:layer-norm-eps 0)))
    (check-exn exn:fail:contract? (lambda () (TransformerEncoderLayer 8)))
    (check-exn (message-matching #rx"#:heads divides the model width")
               (lambda () (TransformerEncoder 8 #:heads 3 #:layers 2)))
    (check-exn blames-this-test
               (lambda () (TransformerDecoder 8 #:heads 3 #:layers 2)))
    (check-exn #rx"^TransformerEncoder: contract violation"
               (lambda () (TransformerEncoder 8 #:heads 2 #:layers 0)))
    (check-exn blames-this-test
               (lambda () (TransformerDecoder 8 #:heads 2 #:layers 2
                                              #:norm 'yes)))
    (check-exn blames-this-test
               (lambda () (TransformerEncoder 8 #:heads 2 #:layers 2
                                              #:activation 'tanh)))
    (check-exn exn:fail:contract?
               (lambda () (TransformerEncoder 8 #:heads 2)))
    (check-exn (message-matching #rx"a model width needs #:heads")
               (lambda () (TransformerEncoder 8 #:layers 2)))
    (define layer (TransformerEncoderLayer 8 #:heads 2))
    (define (make) (TransformerEncoderLayer 8 #:heads 2))
    (define keywords-alone
      (message-matching #rx"the layer keywords configure layers built from"))
    (check-exn keywords-alone
               (lambda () (TransformerEncoder layer #:layers 2 #:heads 2)))
    (check-exn keywords-alone
               (lambda () (TransformerEncoder make #:layers 2 #:bias? #f)))
    (check-exn blames-this-test
               (lambda () (TransformerEncoder layer #:layers 2 #:dropout 0.0)))
    (check-exn (message-matching #rx"#:norm #t makes a LayerNorm")
               (lambda () (TransformerEncoder layer #:layers 2 #:norm #t)))
    (check-exn blames-this-test
               (lambda () (TransformerDecoder make #:layers 2 #:norm #t)))
    (check-exn blames-this-test
               (lambda () (TransformerEncoder (lambda () (Linear 8 8))
                                              #:layers 2)))
    (check-exn blames-this-test
               (lambda () (TransformerDecoder make #:layers 2)))
    (check-exn blames-this-test
               (lambda () (TransformerDecoder layer #:layers 2)))
    (check-exn blames-this-test
               (lambda () (TransformerEncoder (Linear 8 8) #:layers 2)))
    (check-exn #rx"^TransformerEncoder: contract violation"
               (lambda () (TransformerEncoder make #:layers 0)))
    (check-exn (message-matching #rx"expected: even-width")
               (lambda () (sinusoidal-positions 4 7)))
    (check-exn (message-matching #rx"expected: .*position-vector")
               (lambda () (sinusoidal-positions (zeros 2 2) 8)))
    (check-exn (message-matching #rx"#:device places a length")
               (lambda () (sinusoidal-positions (tensor '(1 2)) 8
                                                #:device 'cpu)))
    (check-exn blames-this-test (lambda () (causal-mask 3 #:dtype 'int64))))

  (test-case "an application the layer cannot take is the caller's violation"
    (define encoder (encoder-layer))
    (define decoder (decoder-layer))
    (define tgt (randn 4 2 8))
    (define (refuses who pattern thunk)
      (check-exn (message-matching pattern) thunk)
      (check-exn (message-matching
                  (regexp (format "^~a: contract violation" who)))
                 thunk)
      (check-exn (message-matching #rx"blaming: caller") thunk))
    (refuses 'TransformerEncoderLayer #rx"src is 6 wide, not 8"
             (lambda () (encoder (randn 5 2 6))))
    (refuses 'TransformerEncoderLayer #rx"expected: attention-sequence"
             (lambda () (encoder (randn 1 5 2 8))))
    (refuses 'TransformerEncoderLayer
             #rx"mask has shape [(]5 4[)], not [(]5 5[)] or [(]4 5 5[)]"
             (lambda () (encoder x #:mask (zeros 5 4))))
    (refuses 'TransformerEncoderLayer
             #rx"key-padding-mask has shape [(]5 2[)], not [(]2 5[)]"
             (lambda () (encoder x #:key-padding-mask (zeros 5 2))))
    (refuses 'TransformerEncoderLayer #rx"expected: boolean[?]"
             (lambda () (encoder x #:causal? 1)))
    (refuses 'TransformerDecoderLayer #rx"memory is 6 wide, not 8"
             (lambda () (decoder tgt (randn 7 2 6))))
    (refuses 'TransformerDecoderLayer #rx"tgt is 4 wide, not 8"
             (lambda () (decoder (randn 4 2 4) memory)))
    (refuses 'TransformerDecoderLayer #rx"both batched [(]rank 3[)]"
             (lambda () (decoder (select tgt 1 0) memory)))
    (refuses 'TransformerDecoderLayer #rx"tgt has batch 3 and memory 2"
             (lambda () (decoder (randn 4 3 8) memory)))
    (refuses 'TransformerDecoderLayer
             #rx"tgt-mask has shape [(]4 7[)], not [(]4 4[)] or [(]4 4 4[)]"
             (lambda () (decoder tgt memory #:tgt-mask (zeros 4 7))))
    (refuses 'TransformerDecoderLayer
             #rx"memory-mask has shape [(]4 4[)], not [(]4 7[)] or [(]4 4 7[)]"
             (lambda () (decoder tgt memory #:memory-mask (zeros 4 4))))
    (refuses 'TransformerDecoderLayer
             #rx"tgt-key-padding-mask has shape [(]2 7[)], not [(]2 4[)]"
             (lambda ()
               (decoder tgt memory #:tgt-key-padding-mask (zeros 2 7))))
    (refuses 'TransformerDecoderLayer
             #rx"memory-key-padding-mask has shape [(]2 4[)], not [(]2 7[)]"
             (lambda ()
               (decoder tgt memory #:memory-key-padding-mask (zeros 2 4))))
    (check-exn #rx"TransformerEncoderLayer: arity mismatch"
               (lambda () (encoder x x)))
    (check-exn #rx"procedure: TransformerDecoderLayer.*given keyword: #:mask"
               (lambda () (decoder tgt memory #:mask (zeros 4 4)))))

  (when (cuda-available?)
    (test-case "on CUDA the blocks and stacks agree with the CPU"
      (define encoder (encoder-layer #:norm-first? #t #:activation 'gelu))
      (define decoder (decoder-layer))
      (define tgt (randn 4 2 8))
      (define (both e d src tgt mem pad)
        (list (e src #:key-padding-mask pad #:causal? #t)
              (d tgt mem #:tgt-causal? #t)))
      (define on-cpu (both encoder decoder x tgt memory padded))
      (to encoder 'cuda)
      (to decoder 'cuda)
      (define (on-cuda t) (to-device t 'cuda))
      (for ([a (in-list on-cpu)]
            [b (in-list (both encoder decoder (on-cuda x) (on-cuda tgt)
                              (on-cuda memory) (on-cuda padded)))])
        (check-true (close? a (to-device b 'cpu) 1e-4)))
      (check-equal? (device (causal-mask 2 #:device 'cuda))
                    (cuda-device))
      (check-true (close? (sinusoidal-positions 4 8)
                          (to-device (sinusoidal-positions 4 8 #:device 'cuda)
                                     'cpu))))))

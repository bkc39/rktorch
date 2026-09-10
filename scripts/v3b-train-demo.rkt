#lang racket/base

;; One-block char-GPT smoke for the transformer tranche (#22); superseded
;; by the literate 06-gpt example.
;; Run:  nix develop --command racket scripts/v3b-train-demo.rkt

(require (only-in racket/list remove-duplicates take-right)
         torch
         torch/nn)

(define corpus
  (string-append
   "The Nellie, a cruising yawl, swung to her anchor without a flutter of"
   " the sails, and was at rest. The flood had made, the wind was nearly"
   " calm, and being bound down the river, the only thing for it was to"
   " come to and wait for the turn of the tide. The sea-reach of the"
   " Thames stretched before us like the beginning of an interminable"
   " waterway. In the offing the sea and the sky were welded together"
   " without a joint, and in the luminous space the tanned sails of the"
   " barges drifting up with the tide seemed to stand still in red"
   " clusters of canvas sharply peaked, with gleams of varnished sprits."))

(define vocab (sort (remove-duplicates (string->list corpus)) char<?))
(define vocab-size (length vocab))
(define char->id (for/hash ([c (in-list vocab)] [i (in-naturals)]) (values c i)))
(define id->char (list->vector vocab))
(define (encode str) (for/list ([c (in-string str)]) (hash-ref char->id c)))
(define (decode ids) (list->string (for/list ([i (in-list ids)])
                                     (vector-ref id->char i))))

(define block-size 32)
(define (windows ids stride)
  (for/list ([i (in-range 0 (- (length ids) block-size 1) stride)])
    (cons (take (list-tail ids i) block-size)
          (take (list-tail ids (add1 i)) block-size))))

(define embed-dim 32)

(define-layer gpt-mini (embed-dim tok-emb pos-emb ln1 wq wk wv wo ln2 fc1 fc2 head)
  #:init (vocab-size block-size embed-dim)
  (set! tok-emb (Embedding vocab-size embed-dim))
  (set! pos-emb (Embedding block-size embed-dim))
  (set! ln1 (LayerNorm embed-dim))
  (set! wq (Linear embed-dim embed-dim))
  (set! wk (Linear embed-dim embed-dim))
  (set! wv (Linear embed-dim embed-dim))
  (set! wo (Linear embed-dim embed-dim))
  (set! ln2 (LayerNorm embed-dim))
  (set! fc1 (Linear embed-dim (* 4 embed-dim)))
  (set! fc2 (Linear (* 4 embed-dim) embed-dim))
  (set! head (Linear embed-dim vocab-size))
  #:forward (idx)
  (let* ([t (cadr (tensor-shape idx))]
         [pos (to-dtype (arange t) 'int64)]
         [h (add (tok-emb idx) (pos-emb pos))]
         [hn (ln1 h)]
         [scores (div (matmul (wq hn) (transpose (wk hn) 1 2))
                      (sqrt embed-dim))]
         [mask (eq (tril (ones t t)) 0)]
         [att (softmax (masked-fill scores mask -inf.0) 2)]
         [h (add h (wo (matmul att (wv hn))))]
         [h (add h (fc2 (gelu (fc1 (ln2 h)))))])
    (head h)))

(module+ main
  (manual-seed! 0)
  (define ids (encode corpus))
  (define ws (windows ids 4))
  (define batch (length ws))
  (printf "corpus: ~a chars, vocab ~a, ~a windows of ~a\n"
          (length ids) vocab-size batch block-size)

  (define x (to-dtype (tensor (map car ws)) 'int64))
  (define y (to-dtype (tensor (map cdr ws)) 'int64))
  (define y-flat (reshape y (* batch block-size)))

  (define net (gpt-mini vocab-size block-size embed-dim))
  (define opt (adam (parameters net) #:lr 1e-3))

  (define (loss-now)
    (cross-entropy (reshape (net x) (* batch block-size) vocab-size)
                   y-flat))

  (printf "initial loss ~a (ln vocab = ~a)\n"
          (real->decimal-string (item (loss-now)) 4)
          (real->decimal-string (log vocab-size) 4))
  (for ([step (in-range 1 301)])
    (define loss (loss-now))
    (zero-grads! opt)
    (backward! loss)
    (step! opt)
    (when (zero? (modulo step 50))
      (printf "step ~a: loss ~a\n"
              step (real->decimal-string (item loss) 4))))

  (define (generate seed n)
    (in-eval-mode net
      (with-no-grad
        (for/fold ([out (encode seed)]) ([_ (in-range n)])
          (define ctx (take-right out (min block-size (length out))))
          (define t (length ctx))
          (define logits (net (to-dtype (tensor (list ctx)) 'int64)))
          (define last-row (reshape (narrow logits 1 (- t 1) 1) vocab-size))
          (append out (list (inexact->exact (item (argmax last-row)))))))))

  (printf "sample: ~s\n" (decode (generate "The " 120))))

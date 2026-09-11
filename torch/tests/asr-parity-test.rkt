#lang racket/base

;; Needs the `nix develop` python; the model re-declaration MUST stay in
;; sync with examples/racket/07-asr.rkt, as the 05/06 twins do.

(module+ test
  (require rackunit
           (only-in racket/list last)
           "../main.rkt"
           "../nn.rkt"
           (only-in "../audio/functional.rkt" log-mel-spectrogram)
           (only-in "../audio/librispeech.rkt" load-librispeech-fixture)
           (only-in "../data/text.rkt" encode text->vocab)
           "private/python-env.rkt")

  (cond
    [(not (and (python-torch-available?)
               (python-module-available? "torchaudio")))
     (printf "[asr-parity-test] skipped: python3 torch/torchaudio ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (define (sinusoidal-positions t-len n-embd)
       (define half (quotient n-embd 2))
       (define positions (unsqueeze (arange t-len) 1))
       (define freqs
         (exp (mul (arange half) (- (/ (log 10000.0) half)))))
       (define angles (mul positions (unsqueeze freqs 0)))
       (cat (list (sin angles) (cos angles)) 1))
     (define-layer self-attention (n-embd n-head wq wk wv wo)
       #:init (n-embd n-head)
       (set! wq (Linear n-embd n-embd))
       (set! wk (Linear n-embd n-embd))
       (set! wv (Linear n-embd n-embd))
       (set! wo (Linear n-embd n-embd))
       #:forward (x mask)
       (define batch (car (tensor-shape x)))
       (define seq-len (cadr (tensor-shape x)))
       (define head-dim (quotient n-embd n-head))
       (define (split-heads m)
         (transpose (reshape m batch seq-len n-head head-dim) 1 2))
       (define q (split-heads (wq x)))
       (define k (split-heads (wk x)))
       (define v (split-heads (wv x)))
       (define scores (/ (matmul q (transpose k 2 3)) (sqrt head-dim)))
       (define att
         (softmax (if mask (masked-fill scores mask -inf.0) scores) -1))
       (~> (matmul att v) (transpose 1 2) (reshape batch seq-len n-embd) wo))
     (define-layer cross-attention (n-embd n-head wq wk wv wo)
       #:init (n-embd n-head)
       (set! wq (Linear n-embd n-embd))
       (set! wk (Linear n-embd n-embd))
       (set! wv (Linear n-embd n-embd))
       (set! wo (Linear n-embd n-embd))
       #:forward (x memory mask)
       (define batch (car (tensor-shape x)))
       (define seq-len (cadr (tensor-shape x)))
       (define mem-len (cadr (tensor-shape memory)))
       (define head-dim (quotient n-embd n-head))
       (define (split-heads m len)
         (transpose (reshape m batch len n-head head-dim) 1 2))
       (define q (split-heads (wq x) seq-len))
       (define k (split-heads (wk memory) mem-len))
       (define v (split-heads (wv memory) mem-len))
       (define scores (/ (matmul q (transpose k 2 3)) (sqrt head-dim)))
       (define att
         (softmax (if mask (masked-fill scores mask -inf.0) scores) -1))
       (~> (matmul att v) (transpose 1 2) (reshape batch seq-len n-embd) wo))
     (define-layer feed-forward (fc1 fc2)
       #:init (n-embd)
       (set! fc1 (Linear n-embd (* 4 n-embd)))
       (set! fc2 (Linear (* 4 n-embd) n-embd))
       #:forward (x)
       (~> x fc1 gelu fc2))
     (define-layer asr-encoder-block (ln1 attention ln2 mlp)
       #:init (n-embd n-head)
       (set! ln1 (LayerNorm n-embd))
       (set! attention (self-attention n-embd n-head))
       (set! ln2 (LayerNorm n-embd))
       (set! mlp (feed-forward n-embd))
       #:forward (x mask)
       (define x1 (+ x (attention (ln1 x) mask)))
       (+ x1 (mlp (ln2 x1))))
     (define-layer asr-decoder-block (ln1 attention ln2 cross ln3 mlp)
       #:init (n-embd n-head)
       (set! ln1 (LayerNorm n-embd))
       (set! attention (self-attention n-embd n-head))
       (set! ln2 (LayerNorm n-embd))
       (set! cross (cross-attention n-embd n-head))
       (set! ln3 (LayerNorm n-embd))
       (set! mlp (feed-forward n-embd))
       #:forward (x memory mem-mask)
       (with-default-device (tensor-device x)
         (define seq-len (cadr (tensor-shape x)))
         (define causal (eq (tril (ones seq-len seq-len)) 0))
         (define x1 (+ x (attention (ln1 x) causal)))
         (define x2 (+ x1 (cross (ln2 x1) memory mem-mask)))
         (+ x2 (mlp (ln3 x2)))))
     (define-layer asr (n-embd
                        conv1 conv2 dilations
                        encoders ln-enc ctc-head
                        tok-emb decoders ln-dec head)
       #:init (n-mels vocab-size
               #:n-embd [n-embd 64]
               #:n-head [n-head 4])
       (set! conv1 (Conv1d n-mels n-embd 3 #:stride 2 #:padding 1))
       (set! conv2 (Conv1d n-embd n-embd 3 #:stride 2 #:padding 1))
       (set! dilations
             (LayerList (for/list ([d '(1 2 4 8)])
                          (Conv1d n-embd n-embd 3 #:dilation d #:padding d))))
       (set! encoders
             (LayerList (for/list ([_ (in-range 6)])
                          (asr-encoder-block n-embd n-head))))
       (set! ln-enc (LayerNorm n-embd))
       (set! ctc-head (Linear n-embd (add1 vocab-size)))
       (set! tok-emb (Embedding (+ vocab-size 2) n-embd))
       (set! decoders
             (LayerList (for/list ([_ (in-range 6)])
                          (asr-decoder-block n-embd n-head))))
       (set! ln-dec (LayerNorm n-embd))
       (set! head (Linear n-embd (add1 vocab-size)))
       #:forward (x dec-in lengths)
       (with-default-device (tensor-device x)
         (define (halve n) (quotient (add1 n) 2))
         (define t1 (halve (caddr (tensor-shape x))))
         (define t2 (halve t1))
         (define l1 (and lengths (map halve lengths)))
         (define l2 (and l1 (map halve l1)))
         (define (keep lens t)
           (reshape (to-dtype
                     (lt (unsqueeze (arange t) 0)
                         (unsqueeze (tensor (map exact->inexact lens)) 1))
                     'float32)
                    (length lens) 1 t))
         (define (clip v lens t) (if lens (mul v (keep lens t)) v))
         (define c (clip (relu (conv1 x)) l1 t1))
         (define c0 (clip (relu (conv2 c)) l2 t2))
         (define c4
           (for/fold ([h c0]) ([dil (in-layers dilations)])
             (clip (+ h (relu (dil h))) l2 t2)))
         (define enc-mask
           (and l2
                (reshape (ge (unsqueeze (arange t2) 0)
                             (unsqueeze (tensor (map exact->inexact l2)) 1))
                         (length l2) 1 1 t2)))
         (define t-len (caddr (tensor-shape c4)))
         (define e0 (add (transpose c4 1 2)
                         (sinusoidal-positions t-len n-embd)))
         (define memory
           (ln-enc (for/fold ([h e0]) ([enc (in-layers encoders)])
                     (enc h enc-mask))))
         (define ctc-log-probs (log-softmax (ctc-head memory) 2))
         (define s-len (cadr (tensor-shape dec-in)))
         (define d0 (add (tok-emb dec-in)
                         (sinusoidal-positions s-len n-embd)))
         (define d
           (ln-dec (for/fold ([h d0]) ([dec (in-layers decoders)])
                     (dec h memory enc-mask))))
         (values ctc-log-probs (head d))))
     (define-values (samples rate transcript) (load-librispeech-fixture))
     (define vocab (text->vocab transcript))
     (define v-size (vector-length vocab))
     (let ([shapes (map tensor-shape (parameters (asr 80 v-size)))])
       (check-equal? (length shapes) 273
                     "asr parameter count must match 07-asr.rkt")
       (check-equal? (car shapes) '(64 80 3))
       (check-equal? (list-ref shapes 10) '(64 64 3))
       (check-not-false (member (list (+ v-size 2) 64) shapes))
       (check-equal? (last shapes) (list (add1 v-size))))
     (define char-ids
       (map inexact->exact (tensor->list (encode vocab transcript))))
     (define (train-on device)
       (with-default-device device
         (manual-seed! 0)
         (define x
           (to-device (unsqueeze (log-mel-spectrogram (ref samples 0)
                                                      #:sample-rate rate)
                                 0)
                      device))
         (define targets (unsqueeze (encode vocab transcript) 0))
         (define dec-in
           (unsqueeze (to-dtype (tensor (cons (add1 v-size) char-ids))
                                'int64)
                      0))
         (define dec-out
           (unsqueeze (to-dtype (tensor (append char-ids (list v-size)))
                                'int64)
                      0))
         (define net (asr 80 v-size))
         (define opt (adam (parameters net) #:lr 0.001))
         (define losses
           (for/list ([_ (in-range 5)])
             (zero-grads! opt)
             (define-values (ctc-lp logits) (net x dec-in #f))
             (define loss-ctc
               (ctc-loss (transpose ctc-lp 0 1) targets
                         #:input-lengths
                         (list (cadr (tensor-shape ctc-lp)))
                         #:target-lengths
                         (list (string-length transcript))
                         #:blank v-size
                         #:zero-infinity? #t))
             (define loss-ce
               (cross-entropy (reshape logits -1 (add1 v-size))
                              (reshape dec-out -1)))
             (define loss
               (add (mul 0.3 loss-ctc) (mul 0.7 loss-ce)))
             (backward! loss)
             (step! opt)
             (item loss)))
         (values losses
                 (cat (for/list ([p (in-list (parameters net))])
                        (reshape p -1))))))
     ;; 2e-4, not tol: this backward chain accumulates more
     ;; libtorch-bin-vs-wheel float32 divergence than 05/06 do
     (check-training-twin "07_asr" "python/07_asr.py" train-on 'cpu 2e-4)
     (when (and (cuda-available?)
                (python-cuda-available?))
       (check-training-twin "07_asr" "python/07_asr.py" train-on
                            'cuda 5e-3))]))

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
     (define (blocks make-stack make-layer n-embd n-head)
       (make-stack (lambda ()
                     (make-layer n-embd
                                 #:heads n-head
                                 #:ffn-width (* 4 n-embd)
                                 #:dropout 0.0
                                 #:activation 'gelu
                                 #:norm-first? #t
                                 #:batch-first? #t))
                   #:layers 6
                   #:norm (LayerNorm n-embd)
                   #:copies? #f))
     (define-layer asr (n-embd conv1 conv2 dilations encoder ctc-head
                        tok-emb decoder head)
       #:init (n-mels vocab-size
               #:n-embd [n-embd 64]
               #:n-head [n-head 4])
       (set! conv1 (Conv1d n-mels n-embd 3 #:stride 2 #:padding 1))
       (set! conv2 (Conv1d n-embd n-embd 3 #:stride 2 #:padding 1))
       (set! dilations
             (LayerList (for/list ([d '(1 2 4 8)])
                          (Conv1d n-embd n-embd 3 #:dilation d #:padding d))))
       (set! encoder
             (blocks TransformerEncoder TransformerEncoderLayer n-embd n-head))
       (set! ctc-head (Linear n-embd (add1 vocab-size)))
       (set! tok-emb (Embedding (+ vocab-size 2) n-embd))
       (set! decoder
             (blocks TransformerDecoder TransformerDecoderLayer n-embd n-head))
       (set! head (Linear n-embd (add1 vocab-size)))
       #:forward (x dec-in lengths)
       (with-default-device (tensor-device x)
         (define (halve n) (quotient (add1 n) 2))
         (define t1 (halve (caddr (tensor-shape x))))
         (define t2 (halve t1))
         (define l1 (and lengths (map halve lengths)))
         (define l2 (and l1 (map halve l1)))
         (define (row-lengths lens)
           (unsqueeze (tensor (map exact->inexact lens)) 1))
         (define (clip v lens t)
           (if lens
               (mul v (reshape (to-dtype (lt (unsqueeze (arange t) 0)
                                             (row-lengths lens))
                                         'float32)
                               (length lens) 1 t))
               v))
         (define c (clip (relu (conv1 x)) l1 t1))
         (define c0 (clip (relu (conv2 c)) l2 t2))
         (define c4
           (for/fold ([h c0]) ([dil (in-layers dilations)])
             (clip (+ h (relu (dil h))) l2 t2)))
         (define padding
           (and l2 (ge (unsqueeze (arange t2) 0) (row-lengths l2))))
         (define (positioned v)
           (+ v (sinusoidal-positions (cadr (tensor-shape v)) n-embd
                                      #:layout 'halves)))
         (define memory
           (encoder (positioned (transpose c4 1 2))
                    #:key-padding-mask padding))
         (values (log-softmax (ctc-head memory) 2)
                 (head (decoder (positioned (tok-emb dec-in)) memory
                                #:tgt-causal? #t
                                #:memory-key-padding-mask padding)))))
     (define-values (samples rate transcript) (load-librispeech-fixture))
     (define vocab (text->vocab transcript))
     (define v-size (vector-length vocab))
     (let* ([named (named-parameters (asr 80 v-size))]
            [shapes (map (lambda (p) (tensor-shape (cdr p))) named)])
       (check-equal? (length shapes) 273
                     "asr parameter count must match 07-asr.rkt")
       (check-equal? (car shapes) '(64 80 3))
       (check-equal? (list-ref shapes 10) '(64 64 3))
       (check-equal? (map car (list (list-ref named 12) (list-ref named 113)))
                     '("encoder.layers.0.self-attn.query.weight"
                       "decoder.layers.0.self-attn.query.weight"))
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
     ;; 2e-3, not tol: Adam's first step divides each gradient by its own
     ;; size, and two of the 782k parameters start with gradients near
     ;; 1e-8, Adam's eps, where the libtorch-bin-vs-wheel last bits move
     ;; that step by most of lr; every other parameter agrees within 2.2e-4
     (check-training-twin "07_asr" "python/07_asr.py" train-on 'cpu 2e-3)
     (when (and (cuda-available?)
                (python-cuda-available?))
       (check-training-twin "07_asr" "python/07_asr.py" train-on
                            'cuda 5e-3))]))

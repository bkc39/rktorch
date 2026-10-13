#lang racket/base

;; Needs the `nix develop` python; the model re-declaration MUST stay in
;; sync with torch/examples/07-asr.rkt, as the 05/06 twins do.

(module+ test
  (require (only-in racket/list last)
           rackunit
           (only-in "../audio/functional.rkt" log-mel-spectrogram)
           (only-in "../audio/librispeech.rkt" load-librispeech-fixture)
           "../main.rkt"
           "../nn.rkt"
           (only-in "../text/data.rkt" encode text->vocab)
           "private/python-env.rkt")

  (cond
    [(not (and (python-torch-available?)
               (python-module-available? "torchaudio")))
     (printf "[asr-parity-test] skipped: python3 torch/torchaudio ~a\n"
             "not available (run inside `nix develop`)")]
    [else
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
       (set! encoder (TransformerEncoder n-embd
                                         #:heads n-head
                                         #:layers 6
                                         #:ffn-width (* 4 n-embd)
                                         #:activation 'gelu
                                         #:norm-first? #t
                                         #:dropout 0.0
                                         #:batch-first? #t
                                         #:norm #t))
       (set! ctc-head (Linear n-embd (add1 vocab-size)))
       (set! tok-emb (Embedding (+ vocab-size 2) n-embd))
       (set! decoder (TransformerDecoder n-embd
                                         #:heads n-head
                                         #:layers 6
                                         #:ffn-width (* 4 n-embd)
                                         #:activation 'gelu
                                         #:norm-first? #t
                                         #:dropout 0.0
                                         #:batch-first? #t
                                         #:norm #t))
       (set! head (Linear n-embd (add1 vocab-size)))
       #:forward (x dec-in lengths)
       (with-default-device (tensor-device x)
         (define (halve n) (quotient (add1 n) 2))
         (define t1 (~> (tensor-shape x) caddr halve))
         (define t2 (halve t1))
         (define l1 (and lengths (map halve lengths)))
         (define l2 (and l1 (map halve l1)))
         (define (row-lengths lens)
           (~> (map exact->inexact lens) tensor (unsqueeze 1)))
         (define (clip v lens t)
           (if lens
               (* v (reshape (to-dtype (< (unsqueeze (arange t) 0)
                                          (row-lengths lens))
                                       'float32)
                             (length lens) 1 t))
               v))
         (define c (~> (conv1 x) relu (clip l1 t1)))
         (define c0 (~> (conv2 c) relu (clip l2 t2)))
         (define c4
           (for/fold ([h c0]) ([dil (in-layers dilations)])
             (~> (dil h) relu (+ h) (clip l2 t2))))
         (define padding
           (and l2 (>= (unsqueeze (arange t2) 0) (row-lengths l2))))
         (define (positioned v)
           (~> (shape v)
               cadr
               (sinusoidal-positions n-embd #:layout 'halves)
               (+ v)))
         (define memory
           (~> (transpose c4 1 2)
               positioned
               (encoder #:key-padding-mask padding)))
         (values (log-softmax (ctc-head memory) 2)
                 (~> (tok-emb dec-in)
                     positioned
                     (decoder memory
                              #:tgt-causal? #t
                              #:memory-key-padding-mask padding)
                     head))))
     (define-values (samples rate transcript) (load-librispeech-fixture))
     (define vocab (text->vocab transcript))
     (define v-size (vector-length vocab))
     (let* ([named (named-parameters (asr 80 v-size))]
            [shapes (map (lambda (p) (shape (cdr p))) named)])
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
               (+ (* 0.3 loss-ctc) (* 0.7 loss-ce)))
             (backward! loss)
             (step! opt)
             (item loss)))
         (values losses (named-parameters net))))
     ;; Every parameter is held to the tolerance but the convolution
     ;; weights of the ReLU front end, the only ones that drift past it: 78
     ;; of their 64,512 on the CPU (at most 1.3e-3), none on CUDA, 20 of
     ;; the 78 in channels whose gradient is exactly zero at some step. A
     ;; unit near its kink switches between the two builds' last bits, and
     ;; Adam's step, divided by the gradient's own size, carries the switch
     ;; at up to lr. Every other parameter agrees within half the tolerance.
     (define front-end-weights
       '("conv1.weight" "conv2.weight" "dilations.0.weight"
         "dilations.1.weight" "dilations.2.weight" "dilations.3.weight"))
     (define (check-twin device dev-tol front-end-tol)
       (define j
         (call-with-python-env
          #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
          (lambda () (python-result "python/07_asr.py"))))
       (define-values (losses named) (train-on device))
       (for ([r (in-list losses)]
             [p (in-list (hash-ref j 'losses))]
             [i (in-naturals)])
         (check-= r p dev-tol (format "07_asr[~a]: loss at step ~a" device i)))
       (define theirs (list->vector (hash-ref j 'values)))
       (check-equal? (vector-length theirs)
                     (for/sum ([np (in-list named)]) (numel (cdr np)))
                     (format "07_asr[~a]: parameter count" device))
       (define-values (_end drifted front-end-size)
         (for/fold ([start 0] [drifted 0] [front-end-size 0])
                   ([np (in-list named)])
           (define front-end? (and (member (car np) front-end-weights) #t))
           (define mine
             (~> (cdr np) (to-device 'cpu) (reshape -1) tensor->list))
           (define gaps
             (for/list ([r (in-list mine)] [i (in-naturals start)])
               (define gap (abs (- r (vector-ref theirs i))))
               (check-true (<= gap (if front-end? front-end-tol dev-tol))
                           (format "07_asr[~a]: ~a element ~a off by ~a"
                                   device (car np) (- i start) gap))
               gap))
           (values (+ start (length mine))
                   (+ drifted
                      (if front-end?
                          (for/sum ([gap (in-list gaps)])
                            (if (> gap dev-tol) 1 0))
                          0))
                   (+ front-end-size (if front-end? (length mine) 0)))))
       (check-true (<= drifted (* 0.005 front-end-size))
                   (format "07_asr[~a]: ~a of ~a front-end weights past ~a"
                           device drifted front-end-size dev-tol)))
     (check-twin 'cpu 2e-4 2e-3)
     (when (and (cuda-available?)
                (python-cuda-available?))
       (check-twin 'cuda 5e-3 1e-2))]))

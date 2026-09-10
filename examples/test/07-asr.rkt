#lang racket/base

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         torch
         torch/nn
         (only-in torch/audio/librispeech
                  librispeech-utterances load-librispeech-fixture)
         (only-in torch/audio/metrics cer wer)
         "../racket/07-asr.rkt")

(module+ main
  (define (env-number name default)
    (define v (getenv name))
    (cond [(not v) default]
          [(string->number v) => values]
          [else (error '07-asr "~a is not a number: ~a" name v)]))
  (printf "device: ~a\n" (pick-device))
  (define-values (net vocab)
    (train-librispeech #:epochs (env-number "EPOCHS" 20)
                       #:limit (env-number "LIMIT" #f)))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define hypothesis (transcribe net vocab features))
  (printf "ref:     ~a\nctc:     ~a\nattend:  ~a\nwer: ~a  cer: ~a\n"
          transcript (greedy-decode net vocab features) hypothesis
          (wer transcript hypothesis) (cer transcript hypothesis)))

(module+ test
  (require rackunit)
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 273)
  (check-equal? (first names) "conv1.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "dil4.bias" names))
  (check-not-false (member "enc6.wq.weight" names))
  (check-not-false (member "tok-emb.weight" names))
  (check-not-false (member "dec6.co.bias" names))
  (check-equal? (tensor-shape (car (parameters net))) '(64 80 3))
  (check-equal? (tensor-shape
                 (cdr (assoc "tok-emb.weight" (named-parameters net))))
                (list (+ (vector-length vocab) 2) 64))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define ctc-hyp (greedy-decode net vocab features))
  (define att-hyp (transcribe net vocab features))
  (for ([hypothesis (in-list (list ctc-hyp att-hyp))])
    (check-true (string? hypothesis))
    (check-true (for/and ([c (in-string hypothesis)])
                  (and (member c (vector->list vocab)) #t))
                (format "decoded chars outside the vocab: ~v" hypothesis)))
  (check-true (layer-training? net) "decoding left the net in eval mode")
  (check-equal? (wer transcript transcript) 0)
  (check-equal? (cer transcript transcript) 0)
  (check-equal? (wer transcript "") 1)
  (check-equal? (cer transcript "") 1)
  (define mel (ref features 0))
  (define batch-loss
    (hybrid-batch-loss net vocab
                       (list mel (narrow mel 1 0 200))
                       (list transcript "MISTER QUILTER")))
  (check-true (rational? (item batch-loss)))
  (check-false (nan? (item batch-loss)))
  ;; the masking invariant the conv-stack fix restored: a row's encoder
  ;; output must not depend on how much padding its bucket neighbour forced
  (let ()
    (define short (narrow mel 1 0 200))
    (define t-max (cadr (tensor-shape mel)))
    (define sos-in
      (unsqueeze (to-dtype (tensor (list (add1 (vector-length vocab))))
                           'int64)
                 0))
    (define-values (alone _al) (net (unsqueeze short 0) sos-in #f))
    (define-values (batched _bl)
      (net (stack (list (cat (list short (zeros 80 (- t-max 200))) 1) mel) 0)
           (cat (list sos-in sos-in) 0)
           (list 200 t-max)))
    (define frames (cadr (tensor-shape alone)))
    (for ([a (in-list (tensor->list (ref alone 0)))]
          [b (in-list (tensor->list (narrow (ref batched 0) 0 0 frames)))]
          [i (in-naturals)])
      (check-= a b 1e-4
               (format "padding perturbed encoder output ~a" i))))
  (check-true (<= (string-length att-hyp) 500)
              (format "transcribe failed to terminate: ~v" att-hyp))
  (check-true (device? (pick-device)))
  (let ()
    (manual-seed! 0)
    (define drop-net (asr 80 (vector-length vocab) #:dropout 0.5))
    (define sos-in
      (unsqueeze (to-dtype (tensor (list (add1 (vector-length vocab))))
                           'int64)
                 0))
    (define (logits)
      (define-values (_ctc l) (drop-net features sos-in #f))
      (tensor->list l))
    (check-not-equal? (logits) (logits)
                      "dropout did not perturb a training-mode forward")
    (in-eval-mode drop-net
      (check-equal? (logits) (logits)
                    "dropout stayed active in eval mode")))
  (check-exn #rx"no utterances to score"
             (lambda () (evaluate net vocab '())))
  ;; device RNG streams differ, so this arm checks convergence, never
  ;; equality; on MPS it also crosses ctc-loss's CPU carve-out
  (define accel (pick-device))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net a-vocab _a-dev)
      (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l))))
                        a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))
    (check-true (string? (greedy-decode a-net a-vocab features)))
    (check-true (string? (transcribe a-net a-vocab features)))
    ;; a multi-row batch must come back on the accelerator: ctc-loss's CPU
    ;; carve-out is only correct if it does not strand the graph there
    (define a-mel (to-device mel accel))
    (define a-batch-loss
      (hybrid-batch-loss a-net a-vocab
                         (list a-mel (narrow a-mel 1 0 200))
                         (list transcript "MISTER QUILTER")))
    (check-equal? (tensor-device a-batch-loss) accel)
    (check-true (rational? (item a-batch-loss)))
    (check-false (nan? (item a-batch-loss)))))

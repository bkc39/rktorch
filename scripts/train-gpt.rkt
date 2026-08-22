#lang racket/base

;; Train the 06-gpt example on Heart of Darkness, checkpoint, sample.
;; Run (GPU):  nix develop .#cuda --command racket scripts/train-gpt.rkt
;; Run (CPU):  nix develop --command racket scripts/train-gpt.rkt

(require (only-in racket/file make-directory*)
         (only-in racket/format ~r)
         (only-in racket/path path-only)
         torch
         torch/nn
         (only-in torch/data/text
                  contiguous-blocks
                  encode
                  load-heart-of-darkness
                  text->vocab)
         (only-in "../examples/racket/06-gpt.rkt" generate gpt pick-device))

(define epochs (string->number (or (getenv "EPOCHS") "40")))
(define batch (string->number (or (getenv "BATCH") "64")))
(define block-size (string->number (or (getenv "BLOCK") "64")))
(define n-embd 128)
(define n-head 4)
(define n-layer 4)
(define prompts '("The " "Marlow " "The river "))
(define checkpoint (or (getenv "CHECKPOINT") "checkpoints/gpt-hod"))

(define device (pick-device))
(printf "device: ~a\n" device)

(with-default-device device
  (manual-seed! 0)
  (define text (load-heart-of-darkness))
  (define vocab (text->vocab text))
  (define v-size (vector-length vocab))
  (define-values (xs ys) (contiguous-blocks (encode vocab text) block-size))
  (define n (car (tensor-shape xs)))
  (unless (<= batch n)
    (error 'train-gpt "batch ~a exceeds the corpus's ~a blocks" batch n))
  (printf "corpus: ~a chars, vocab ~a; ~a blocks of ~a (~a minibatches/epoch)\n"
          (string-length text) v-size n block-size (quotient n batch))
  (define net (gpt v-size block-size
                   #:n-embd n-embd #:n-head n-head #:n-layer n-layer))
  (define opt (adam (parameters net) #:lr 0.0003))
  (for ([epoch (in-range 1 (add1 epochs))])
    (define-values (total steps)
      ;; add1: include the final window at n - batch (batch = n must run
      ;; one full-batch step, not zero)
      (for/fold ([total 0.0] [steps 0])
                ([start (in-range 0 (add1 (- n batch)) batch)])
        (zero-grads! opt)
        (define loss
          (cross-entropy
           (reshape (net (narrow xs 0 start batch)) -1 v-size)
           (reshape (narrow ys 0 start batch) -1)))
        (backward! loss)
        (step! opt)
        (values (+ total (item loss)) (add1 steps))))
    (printf "epoch ~a/~a: mean loss ~a\n"
            epoch epochs (~r (/ total steps) #:precision '(= 4)))
    (flush-output))
  ;; return the training high-water's cached VRAM before generation
  (reclaim-native-memory!)
  (let ([dir (path-only (string->path checkpoint))])
    (when dir (make-directory* dir)))
  (save-state! net (string-append checkpoint ".safetensors"))
  (call-with-output-file (string-append checkpoint ".rktd") #:exists 'replace
    (lambda (out)
      (write (list (cons 'vocab (list->string (vector->list vocab)))
                   (cons 'block-size block-size)
                   (cons 'n-embd n-embd)
                   (cons 'n-head n-head)
                   (cons 'n-layer n-layer)
                   (cons 'epochs epochs))
             out)))
  (printf "\ncheckpoint: ~a.safetensors (+ .rktd sidecar)\n" checkpoint)
  (for ([prompt (in-list prompts)])
    (printf "\n--- prompt ~v ---\n~a\n" prompt
            (generate net vocab prompt #:steps 300))))

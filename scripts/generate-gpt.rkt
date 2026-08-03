#lang racket/base

;; Load a scripts/train-gpt.rkt checkpoint and run autoregressive greedy
;; rollouts — no training, just forward passes, so it runs comfortably on
;; CPU (or DEVICE=cuda).
;;
;; Usage:
;;   racket scripts/generate-gpt.rkt [prefix] [prompt ...]
;;
;; `prefix` defaults to checkpoints/gpt-hod (the train script's default);
;; <prefix>.safetensors holds the weights, <prefix>.rktd the vocab +
;; architecture sidecar. Each prompt gets a STEPS-char rollout (default
;; 300). Prompts must use only characters from the training vocab.

(require torch
         torch/nn
         (only-in "../examples/racket/06-gpt.rkt" generate gpt))

(define args (vector->list (current-command-line-arguments)))
(define prefix (if (pair? args) (car args) "checkpoints/gpt-hod"))
(define prompts
  (if (and (pair? args) (pair? (cdr args))) (cdr args) '("The ")))
(define steps (string->number (or (getenv "STEPS") "300")))
(define device (string->symbol (or (getenv "DEVICE") "cpu")))

(define config
  (with-input-from-file (string-append prefix ".rktd") read))
(define (cfg key) (cdr (assq key config)))
(define vocab (list->vector (string->list (cfg 'vocab))))

(printf "checkpoint ~a (~a epochs trained); vocab ~a; device ~a\n"
        prefix (cfg 'epochs) (vector-length vocab) device)

(with-default-device device
  ;; The seeded init is immediately overwritten by load-state!; seeding just
  ;; keeps construction deterministic.
  (manual-seed! 0)
  (define net (gpt (vector-length vocab) (cfg 'block-size)
                   #:n-embd (cfg 'n-embd)
                   #:n-head (cfg 'n-head)
                   #:n-layer (cfg 'n-layer)))
  (load-state! net (string-append prefix ".safetensors"))
  (for ([prompt (in-list prompts)])
    (printf "\n--- prompt ~v ---\n~a\n" prompt
            (generate net vocab prompt
                      #:steps steps
                      #:block-size (cfg 'block-size)
                      #:device device))))

#lang info

;; raco review lints this as a normal module and flags every `info` definition
;; as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

(define test-timeouts
  '(("00-randn.rkt" 60)
    ("01-arith.rkt" 60)
    ("02-matmul.rkt" 60)
    ("03-autograd.rkt" 60)
    ("04-mlp.rkt" 60)
    ("05-mnist.rkt" 60)
    ("06-gpt.rkt" 60)
    ("07-asr.rkt" 90)
    ("07-train-asr.rkt" 60)
    ("08-diffusion.rkt" 60)
    ("09-resnet.rkt" 60)
    ("10-dcgan.rkt" 60)
    ("11-vae.rkt" 60)
    ("12-char-rnn.rkt" 60)
    ("13-translation.rkt" 60)
    ("14-imagenet.rkt" 60)
    ("15-finetune.rkt" 60)
    ("16-style-transfer.rkt" 60)
    ("17-workers.rkt" 60)
    ("18-vit.rkt" 120)))

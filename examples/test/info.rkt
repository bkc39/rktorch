#lang info

;; raco review lints this as a normal module and flags every `info` definition
;; as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

(define test-timeouts
  '(("00-randn.rkt" 300)
    ("01-arith.rkt" 300)
    ("02-matmul.rkt" 300)
    ("03-autograd.rkt" 300)
    ("04-mlp.rkt" 300)
    ("05-mnist.rkt" 300)
    ("06-gpt.rkt" 300)
    ("07-asr.rkt" 300)
    ("08-diffusion.rkt" 600)
    ("09-resnet.rkt" 600)
    ("10-dcgan.rkt" 600)
    ("11-vae.rkt" 600)
    ("12-char-rnn.rkt" 300)
    ("13-translation.rkt" 600)
    ("14-imagenet.rkt" 300)
    ("15-finetune.rkt" 600)))

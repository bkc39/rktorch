#lang info

#|review: ignore|#

(define collection "torch-native")
(define version "0.1")
(define deps
  '("base"
    ("torch-cuda-libs-x86_64-linux" #:version "0.1")
    ("torch-cudnn-x86_64-linux" #:version "0.1")))
(define pkg-desc "K0 spike: libtorchrkt and libtorch 2.14.0 (CUDA 13.0) for torch on x86_64 Linux")
(define pkg-authors (list (string-append "bkschemer" "@" "gmail" "." "com")))

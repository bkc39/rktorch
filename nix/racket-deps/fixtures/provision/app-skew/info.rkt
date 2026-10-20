#lang info
(define collection "rdfix-app")
(define version "1.1")
(define deps '("base" "rdfix-a"
               ("torch-x86_64-linux" #:platform "x86_64-linux" #:version "1.1")
               ("torch-x86_64-linux" #:platform "x86_64-linux-natipkg" #:version "1.1")
               ("torch-aarch64-macosx" #:platform "aarch64-darwin" #:version "1.1")
               ("torch-aarch64-macosx" #:platform "aarch64-macosx" #:version "1.1")))

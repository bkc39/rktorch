#lang info
(define version "0.2")
(define deps '("base" "threading-lib"
               ("torch-x86_64-linux" #:platform "x86_64-linux" #:version "0.1")
               ("torch-aarch64-macosx" #:platform "aarch64-macosx" #:version "0.1")))
(define build-deps '("rackunit-lib" "threading-doc"))

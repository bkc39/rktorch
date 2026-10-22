#lang info
(define version "0.1")
(define native-version "0.1")
(define deps '("base" "threading-lib"
               ("torch-x86_64-linux" #:platform "aarch64-macosx" #:version "0.1")))
(define build-deps '("rackunit-lib" "threading-doc"))

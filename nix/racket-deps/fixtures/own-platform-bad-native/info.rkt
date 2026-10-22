#lang info
(define version "0.1")
(define native-version "0.1-alpha")
(define deps '("base" "threading-lib"
               ("torch-x86_64-linux" #:platform "x86_64-linux" #:version "0.1")))
(define build-deps '("rackunit-lib" "threading-doc"))

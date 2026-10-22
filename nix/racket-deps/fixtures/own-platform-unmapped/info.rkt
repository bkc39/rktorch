#lang info
(define version "0.1")
(define native-version "0.1")
(define deps '("base" "threading-lib"
               ("torch-aarch64-macosx" #:platform "x86_64-macosx" #:version "0.1")))
(define build-deps '("rackunit-lib" "threading-doc"))

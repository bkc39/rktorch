#lang info
(define deps '("base" "threading-lib" ("https://example.com/foo.git" #:version "1.2")))
(define build-deps '("rackunit-lib" "threading-doc"))

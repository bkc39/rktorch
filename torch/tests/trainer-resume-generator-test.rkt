#lang racket/base

(module+ test
  (require rackunit
           (only-in "private/resume-harness.rkt" check-resume fixture-fit-args fixture-world))

  (test-case "T3-L1: the resume law with a shuffling loader that has its own generator"
    (check-resume (lambda (dir) (fixture-world dir #:generator? #t))
                  #:split '(2) #:total 4 #:fit-args fixture-fit-args)))

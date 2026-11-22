#lang racket/base

(module+ test
  (require rackunit
           (only-in "private/resume-harness.rkt" check-resume fixture-fit-args fixture-world))

  (test-case "T3-L1: 2 epochs, resume, 2 more equals 4 straight"
    (check-resume fixture-world #:split '(2) #:total 4 #:fit-args fixture-fit-args))

  (test-case "T3-L2: 1+1+1+1 resumes equal 4 straight"
    (check-resume fixture-world #:split '(1 2 3) #:total 4 #:fit-args fixture-fit-args)))

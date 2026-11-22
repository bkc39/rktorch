#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" cuda-available? device use-deterministic-algorithms!)
           (only-in "private/resume-harness.rkt" check-resume fixture-fit-args fixture-world)
           (only-in "private/switches.rkt" within/switches-restored))

  (test-case "T3-L1g: the resume law on CUDA, with deterministic algorithms"
    (cond
      [(not (cuda-available?)) (displayln "skipped: no CUDA device")]
      [else
       (within/switches-restored
        540
        (lambda ()
          (use-deterministic-algorithms! #t)
          (check-resume (lambda (dir) (fixture-world dir #:generator? #t))
                        #:split '(2) #:total 4
                        #:fit-args (cons (cons '#:device (device 'cuda)) fixture-fit-args))))])))

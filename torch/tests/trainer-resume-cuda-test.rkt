#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" cuda-available? deterministic-algorithms-enabled?
                    use-deterministic-algorithms!)
           (only-in "private/resume-harness.rkt" check-resume fixture-fit-args fixture-world))

  (test-case "T3-L1g: the resume law on CUDA, with deterministic algorithms"
    (cond
      [(not (cuda-available?)) (printf "skipped: no CUDA device\n")]
      [else
       (define was (deterministic-algorithms-enabled?))
       (dynamic-wind
        (lambda () (use-deterministic-algorithms! #t))
        (lambda ()
          (check-resume (lambda (dir) (fixture-world dir #:generator? #t))
                        #:split 2 #:total 4
                        #:fit-args (cons '(#:device . cuda) fixture-fit-args)))
        (lambda () (use-deterministic-algorithms! was)))])))

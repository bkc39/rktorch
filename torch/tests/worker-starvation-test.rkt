#lang racket/base

(module+ test
  (require rackunit
           (only-in "private/wedge.rkt" exit-status-of))

  ;; Main never yields: each `mul!` is a plain foreign call. A worker's ops
  ;; that hopped to main's OS thread would wait for it and never finish.
  (define starved-worker
    '(begin
       (require (only-in torch + mul! ones)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (call-with-process-deadline
        90
        (lambda ()
          (define t (ones 64 64))
          (define made (box 0))
          (define worker
            (thread #:pool 'own
                    (lambda ()
                      (for ([_ (in-range 2000)])
                        (void (+ (ones 8 8) 1.0))
                        (set-box! made (add1 (unbox made)))))))
          (define deadline (+ (current-inexact-milliseconds) 30000))
          (let loop ()
            (unless (or (thread-dead? worker)
                        (> (current-inexact-milliseconds) deadline))
              (mul! t 1.0)
              (loop)))
          (exit (if (= 2000 (unbox made)) 0 1))))))

  (test-case "a worker's ops finish while main loops plain foreign calls"
    (check-equal? (exit-status-of starved-worker #:within 120) 0
                  "the worker made fewer than 2000 results in 30 s")))

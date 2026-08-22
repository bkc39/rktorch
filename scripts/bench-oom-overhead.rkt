#lang racket/base

;; Overhead of the OOM-kind check + retry wrap: run on master and on the
;; branch, same host, and compare.
;; Run:  racket scripts/bench-oom-overhead.rkt

(require racket/format
         torch
         (only-in "../examples/racket/04-mlp.rkt" [run-example mlp-run])
         (only-in "../examples/racket/05-mnist.rkt" [run-example mnist-run])
         (only-in "../examples/racket/06-gpt.rkt" [run-example gpt-run]))

(define (time-once thunk)
  (collect-garbage)
  (collect-garbage)
  (define t0 (current-inexact-milliseconds))
  (thunk)
  (- (current-inexact-milliseconds) t0))

(define (bench label thunk #:runs [runs 3])
  (define times (for/list ([_ (in-range runs)]) (time-once thunk)))
  (printf "~a: best ~a ms  (runs: ~a)\n"
          label
          (~r (apply min times) #:precision '(= 1))
          (map (lambda (t) (~r t #:precision '(= 1))) times))
  (apply min times))

(module+ main
  (printf "== bench-oom-overhead ==\n")
  (manual-seed! 0)

  ;; 8x8: wrapper overhead, not kernel time, dominates
  (define a (randn 8 8))
  (define b (randn 8 8))
  (define n 200000)
  (define ms (bench "micro: add 8x8 x200k"
                    (lambda () (for ([_ (in-range n)]) (add a b)))))
  (printf "  -> ~a us/op\n" (~r (/ (* ms 1000.0) n) #:precision '(= 3)))

  (bench "04-mlp run-example (cpu)" (lambda () (mlp-run 'cpu)))
  (bench "05-mnist run-example (cpu)"
         (lambda () (mnist-run #:steps 5 #:device 'cpu)))
  (bench "06-gpt run-example (cpu, 20 steps)"
         (lambda () (gpt-run #:steps 20 #:device 'cpu))))

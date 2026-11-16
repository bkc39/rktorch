#lang racket/base

(require (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in (submod "../torch/vision/fashion-mnist.rkt" private)
                  fashion-mnist-idx fixture-files)
         (only-in "../torch/vision/private/idx.rkt" idx-head))

(define-runtime-path fixture-dir "../torch/vision/fixtures/fashion-mnist")

(module+ main
  (require (only-in racket/cmdline command-line)
           (only-in racket/file make-directory*)
           (only-in racket/match match-define))
  (define out
    (command-line #:args ([out-dir #f]) (or out-dir fixture-dir)))
  (make-directory* out)
  (for ([entry (in-list fixture-files)])
    (match-define (list archive name n) entry)
    (call-with-output-file (build-path out name) #:exists 'truncate
      (lambda (port) (write-bytes (idx-head (fashion-mnist-idx archive) n) port)))
    (printf "wrote ~a\n" (build-path out name))))

#lang racket/base

(require (only-in racket/contract/base ->i)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../torch/private/contract.rkt" define/contract-out)
         (only-in "../torch/vision/mnist.rkt" read-idx)
         (only-in (submod "../torch/vision/fashion-mnist.rkt" private) fashion-mnist-idx))

(define (record-count bs)
  (define-values (dims _data) (read-idx bs))
  (car dims))

(define/contract-out (idx-head bs n)
  (->i ([bs bytes?] [n exact-nonnegative-integer?])
       #:pre/name (bs n) "n exceeds the buffer's record count"
       (<= n (record-count bs))
       [_ bytes?])
  (error 'idx-head "unimplemented"))

(define-runtime-path fixture-dir "../torch/vision/fixtures/fashion-mnist")

(define fixture-files
  '(("train-images-idx3-ubyte.gz" "train-512-images-idx3-ubyte" 512)
    ("train-labels-idx1-ubyte.gz" "train-512-labels-idx1-ubyte" 512)
    ("t10k-images-idx3-ubyte.gz" "t10k-128-images-idx3-ubyte" 128)
    ("t10k-labels-idx1-ubyte.gz" "t10k-128-labels-idx1-ubyte" 128)))

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

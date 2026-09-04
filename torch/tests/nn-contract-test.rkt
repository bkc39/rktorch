#lang racket/base

(module layers racket/base
  (require (only-in racket/contract/base -> ->i any/c)
           (only-in "../nn/module.rkt" define-module))

  (define-module AvgPool3d (kernel-size)
    #:contract (-> exact-positive-integer? avg-pool3d?)
    #:forward (x) x)

  (define-module RMSNorm (n)
    #:predicate norm?
    #:contract (-> exact-positive-integer? norm?)
    #:forward (x) x)

  (define-module scale (s)
    #:contract (-> real? scale?)
    #:forward (x) x)

  (define-module GPT2Block (n-embd n-head)
    #:contract (->i ([n-embd exact-positive-integer?]
                     [n-head exact-positive-integer?])
                    #:pre (n-embd n-head) (zero? (remainder n-embd n-head))
                    [_ gpt2-block?])
    #:forward (x) x)

  (define-module Plain (v)
    #:forward (x) x)
  (provide Plain Plain?)) ;; noqa

(module+ test
  (require (only-in racket/contract/base -> any/c)
           rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           (submod ".." layers)
           (only-in "../main.rkt" randn tensor-shape)
           "../nn.rkt")

  (define ((message-matching pattern) e)
    (and (exn:fail:contract? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))
  (define blames-this-test
    (message-matching
     #rx"blaming: [(][^)]*nn-contract-test\\.rkt test[)]"))

  (test-case "a layer's #:contract blames the caller, not torch/nn"
    (check-exn #rx"^Linear: contract violation" (lambda () (Linear 0 3)))
    (check-exn blames-this-test (lambda () (Linear 0 3)))
    (check-exn #rx"^Conv2d: contract violation"
               (lambda () (Conv2d 1 8 3 #:stride 0)))
    (check-exn #rx"^LayerNorm: contract violation"
               (lambda () (LayerNorm '())))
    (check-exn #rx"^Dropout: contract violation"
               (lambda () (Dropout #:p 1)))
    (check-exn #rx"^Sequential: contract violation"
               (lambda () (Sequential (Linear 2 2) 'not-a-module)))
    (check-exn blames-this-test
               (lambda () (Sequential (Linear 2 2) 'not-a-module))))

  (test-case "the exported predicate is the lowercase name"
    (check-true (linear? (Linear 4 3)))
    (check-true (max-pool2d? (MaxPool2d 2)))
    (check-true (avg-pool3d? (AvgPool3d 2)))
    (check-false (avg-pool3d? (Linear 4 3)))
    (check-true (conv2d? (Conv2d 1 1 1)))
    (check-exn #rx"^avg-pool3d[?]: arity mismatch"
               (lambda () (avg-pool3d? 1 2)))
    (check-exn #rx"^linear[?]: arity mismatch"
               (lambda () (linear?))))

  (test-case "#:predicate overrides the derived name; lowercase keeps it"
    (check-true (norm? (RMSNorm 3)))
    (check-true (scale? (scale 1.5)))
    (check-exn #rx"^scale: contract violation" (lambda () (scale "x"))))

  (test-case "->i states the cross-argument invariant a guard used to"
    (check-true (gpt2-block? (GPT2Block 32 4)))
    (check-exn #rx"^GPT2Block: contract violation"
               (lambda () (GPT2Block 32 5)))
    (check-exn #rx"#:pre condition violation"
               (lambda () (GPT2Block 32 5)))
    (check-exn blames-this-test (lambda () (GPT2Block 32 5))))

  (test-case "without #:contract nothing is exported or renamed"
    (check-true (Plain? (Plain 1)))
    (check-equal? (tensor-shape ((Plain 1) (randn 2 3))) '(2 3)))

  (test-case "the plain functions blame the caller too"
    (check-exn #rx"^parameters: contract violation"
               (lambda () (parameters 5)))
    (check-exn blames-this-test (lambda () (parameters 5)))
    (check-exn #rx"^named-parameters: contract violation"
               (lambda () (named-parameters (Linear 2 2) 'prefix)))
    (check-exn #rx"^sgd: contract violation"
               (lambda () (sgd '() #:lr "fast")))
    (check-exn #rx"^adam: contract violation"
               (lambda () (adam (list 1))))
    (check-exn #rx"^step!: contract violation" (lambda () (step! 'opt)))
    (check-exn #rx"^mse-loss: contract violation"
               (lambda () (mse-loss 1 2)))
    (check-exn #rx"^uniform-init: contract violation"
               (lambda () (uniform-init '(2) 0 "one")))
    (check-exn #rx"^kaiming-uniform: contract violation"
               (lambda () (kaiming-uniform '(-1))))
    (check-exn #rx"^state-dict: contract violation"
               (lambda () (state-dict 'model)))
    (check-exn #rx"^save-state!: contract violation"
               (lambda () (save-state! (Linear 2 2) 7)))
    (check-exn #rx"^train!: contract violation" (lambda () (train! 1)))
    (check-exn #rx"^call-with-eval-mode: contract violation"
               (lambda () (call-with-eval-mode (Linear 2 2) 'thunk))))

  (test-case "#:contract outside module level is a syntax error"
    (check-exn #rx"only allowed at module level"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-module Local (v)
                      #:contract (-> any/c local?)
                      #:forward (x) x)
                    (Local 1))))))

  (test-case "#:predicate without #:contract is a syntax error"
    (check-exn #rx"needs #:contract"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-module Local (v)
                      #:predicate local?
                      #:forward (x) x)
                    (Local 1)))))))

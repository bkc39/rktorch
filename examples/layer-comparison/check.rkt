#lang racket/base

(require (only-in json write-json)
         (only-in rackunit check-equal? check-true)
         ;; Whole-module import for define-runtime-path's expansion.
         racket/runtime-path
         (only-in torch
                  add arange backward! cat has-grad? manual-seed! mean mul narrow
                  randn reshape tensor->list tensor-numel tensor-shape with-no-grad)
         (only-in torch/nn
                  adam buffers eval! named-parameters parameters step! train! zero-grads!)
         "models.rkt")

(define-runtime-path reference-path "racket-reference.json")

(define (encoded t)
  (hasheq 'shape (tensor-shape t) 'data (tensor->list t)))

(define (check-model name model x expected)
  (eval! model)
  (define output (model x))
  (check-equal? (tensor-shape output) expected)
  (define target (mul (apply reshape (arange (tensor-numel output)) expected) 0.001))
  (backward! (mean (mul output target)))
  (check-true (andmap has-grad? (parameters model)))
  (define record
    (hasheq 'name name 'input (encoded x) 'output (encoded output)
            'parameters
            (for/list ([p (in-list (named-parameters model))])
              (hasheq 'name (car p) 'tensor (encoded (cdr p))))))
  (define optimizer (adam (parameters model)))
  (zero-grads! optimizer)
  (train! model)
  (define training-output (model x))
  (backward! (mean (mul training-output target)))
  (step! optimizer)
  (eval! model)
  (check-equal? (tensor-shape (with-no-grad (model x))) expected)
  (eprintf "~a: shape ~a, ~a parameter tensors, ~a scalars, backward + Adam OK\n"
           name expected (length (parameters model))
           (apply + (map tensor-numel (parameters model))))
  record)

(manual-seed! 42)
(define vision (SmallResNet))
(define transformer (TransformerStack 32 4 2 16))
(define x (randn 2 8 32))
(void (eval! transformer))
(define baseline (transformer x))
(define changed (cat (list (narrow x 1 0 4) (add (narrow x 1 4 4) 100)) 1))
(define changed-output (transformer changed))
(check-equal? (tensor->list (narrow baseline 1 0 4))
              (tensor->list (narrow changed-output 1 0 4)))
(check-equal? (tensor->list (transformer x)) (tensor->list baseline))
(check-equal? (length (buffers transformer)) 2)
(check-equal? (tensor-shape (transformer (randn 1 3 32))) '(1 3 32))
(eprintf "Transformer: causal isolation, shorter sequences, registered masks, eval determinism OK\n")
(define records
  (list (check-model "resnet" vision (randn 2 3 32 32) '(2 10))
        (check-model "transformer" transformer x '(2 8 32))))
(call-with-output-file reference-path
  (lambda (out) (write-json records out)) #:exists 'truncate/replace)

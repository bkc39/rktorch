#lang racket/base

(module+ test
  (require rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           "../main.rkt"
           "../nn.rkt")

  (define-layer Kinds (n w b shift child absent)
    #:init (n)
    (set! w (Parameter (ones 2)))
    (set! b (Parameter (zeros 2)))
    (set! shift (Buffer (ones 2)))
    (set! child (Linear 2 2))
    #:forward (x)
    (list n w b shift child absent))

  (test-case "fields are classified by value at construction"
    (manual-seed! 0)
    (define k (Kinds 7))
    (check-equal? (map car (named-parameters k))
                  '("w" "b" "child.weight" "child.bias"))
    (check-true (andmap requires-grad? (parameters k)))
    (check-equal? (map tensor-shape (buffers k)) '((2)))
    (check-equal? (map car (named-children k)) '("child"))
    (check-true (linear? (car (children k))))
    (define seen (k (ones 2)))
    (check-equal? (car seen) 7 "a plain field is kept, not registered")
    (check-false (list-ref seen 5) "a field never assigned is #f"))

  (test-case "own parameters come before children's, each in declaration order"
    (define-layer Interleaved (fc1 w fc2 v)
      #:init ()
      (set! fc1 (Linear 1 1))
      (set! w (Parameter (ones 1)))
      (set! fc2 (Linear 1 1))
      (set! v (Parameter (ones 1)))
      #:forward (x) x)
    (check-equal? (map car (named-parameters (Interleaved)))
                  '("w" "v" "fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias")))

  (test-case "a bare tensor is a plain field; only Parameter and Buffer register"
    (define-layer Plain (t)
      #:init ()
      (set! t (ones 3))
      #:forward (x) (add x t))
    (define p (Plain))
    (check-equal? (parameters p) '())
    (check-equal? (buffers p) '())
    (check-equal? (tensor->list (p (zeros 3))) '(1.0 1.0 1.0)))

  (test-case "a field that names an #:init argument starts as that argument"
    (define-layer Scaled (scale w)
      #:init (scale #:width [width 2])
      (set! scale (* 2 scale))
      (set! w (Parameter (ones width)))
      #:forward (x) (mul x scale))
    (define s (Scaled 3))
    (check-equal? (tensor->list (s (ones 1))) '(6.0))
    (check-equal? (map tensor-shape (parameters s)) '((2))))

  (test-case "without #:init the fields are the constructor formals"
    (define-layer Pool (kernel #:stride [stride #f] #:pad [pad 0])
      #:forward (x) (list kernel stride pad))
    (check-equal? ((Pool 2) 'x) '(2 #f 0))
    (check-equal? ((Pool 3 #:stride 1 #:pad 1) 'x) '(3 1 1))
    (check-equal? (parameters (Pool 2)) '()))

  (test-case "#:init takes a rest argument, as #:rest or dotted"
    (define-layer Stack (layers)
      #:init (#:rest ms)
      (set! layers (LayerList ms))
      #:forward (x)
      (for/fold ([acc x]) ([m (in-list (layer-list->list layers))]) (m acc)))
    (define-layer Stack2 (layers)
      #:init (first . rest)
      (set! layers (LayerList (cons first rest) #:prefix "s"))
      #:forward (x) x)
    (manual-seed! 0)
    (define st (Stack (Linear 2 3) (Linear 3 1)))
    (check-equal? (map car (named-parameters st))
                  '("layers.0.weight" "layers.0.bias"
                    "layers.1.weight" "layers.1.bias"))
    (check-equal? (tensor-shape (st (randn 4 2))) '(4 1))
    (check-equal? (map car (named-parameters (Stack2 (Linear 1 1) (Linear 1 1))))
                  '("s.0.weight" "s.0.bias" "s.1.weight" "s.1.bias")))

  (test-case "LayerList: #:prefix \"\" names children by index alone"
    (define-layer Seq (layers)
      #:init (#:rest ms)
      (set! layers (LayerList ms #:prefix ""))
      #:forward (x) x)
    (define s (Seq (Linear 1 1) (Dropout) (Linear 1 1)))
    (check-equal? (map car (named-parameters s))
                  '("0.weight" "0.bias" "2.weight" "2.bias"))
    (check-equal? (length (layer-list->list (car (children s)))) 3)
    (check-true (layer-list? (car (children s))))
    (check-equal? (map car (named-children (car (children s)))) '("0" "1" "2"))
    (check-exn #rx"LayerList: not applicable"
               (lambda () ((car (children s)) (ones 1)))))

  (test-case "LayerList nests under a field and forwards training mode"
    (define-layer Outer (blocks)
      #:init ()
      (set! blocks (Sequential (Linear 1 1) (Dropout #:p 0.5)))
      #:forward (x) (blocks x))
    (define o (Outer))
    (check-equal? (map car (named-parameters o)) '("blocks.0.weight" "blocks.0.bias"))
    (check-true (layer-training? o))
    (eval! o)
    (check-false (layer-training? o))
    (check-false (layer-training? (cadr (layer-list->list
                                         (car (children (car (children o))))))))
    (train! o)
    (check-true (layer-training? o)))

  (test-case "Parameter and Buffer are tensors that keep their identity"
    (define t (ones 2))
    (define p (Parameter t))
    (check-true (tensor? p))
    (check-true (Parameter? p))
    (check-false (Parameter? t))
    (check-false (Buffer? p))
    (check-true (requires-grad? p))
    (check-equal? (tensor->list (add p p)) '(2.0 2.0))
    (define b (Buffer (zeros 2)))
    (check-true (tensor? b))
    (check-true (Buffer? b))
    (check-false (Parameter? b))
    (check-exn #rx"^Parameter: contract violation" (lambda () (Parameter 5)))
    (check-exn #rx"^Buffer: contract violation" (lambda () (Buffer 'x)))
    (check-exn #rx"^LayerList: contract violation"
               (lambda () (LayerList (list 1 2)))))

  (test-case "#:reflection-name and #:init may come in either order"
    (define-layer A (w)
      #:reflection-name 'Renamed
      #:init ()
      (set! w (Parameter (ones 1)))
      #:forward (x) x)
    (define-layer B (w)
      #:init ()
      (set! w (Parameter (ones 1)))
      #:reflection-name 'Renamed
      #:forward (x) x)
    (check-equal? (object-name (A)) 'Renamed)
    (check-equal? (object-name (B)) 'Renamed)
    (check-equal? (map car (named-parameters (B))) '("w")))

  (test-case "with #:init, a field with a default or keyword is a syntax error"
    (check-exn #rx"with #:init, a field is a bare identifier"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-layer Bad ([w #f])
                      #:init ()
                      #:forward (x) x)
                    (Bad)))))
    (check-exn #rx"with #:init, a field is a bare identifier"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-layer Bad (#:w w)
                      #:init ()
                      #:forward (x) x)
                    (Bad)))))))

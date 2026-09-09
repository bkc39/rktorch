#lang racket/base

(require (only-in rackunit check-eq? check-equal? check-exn check-false check-true
                 test-case)
         (only-in "../main.rkt" + backward! has-grad? lambda~> ones relu sum
                  tensor->list zeros)
         (only-in "../nn.rkt" Buffer Dropout LayerList Linear Parameter Sequential
                  buffers children eval! forward layer-forward layer-training?
                  layer? named-parameters parameters procedure->Layer sgd
                  state-dict step! train!))

(module+ test
  (test-case "plain procedures preserve positional arguments and return values"
    (define identity (procedure->Layer values))
    (check-true (layer? identity))
    (check-equal? (call-with-values (lambda () (identity 1 2)) list) '(1 2))
    (check-equal? (call-with-values (lambda () (forward identity)) list) '())
    (check-equal? (layer-forward identity 7) 7)
    (check-equal? (parameters identity) '())
    (check-equal? (buffers identity) '())
    (check-equal? (children identity) '())
    (check-exn #rx"original failure"
               (lambda () ((procedure->Layer (lambda () (error "original failure")))))))

  (test-case "captured parameters and buffers require explicit registration"
    (define weight (Parameter (ones 2)))
    (define offset (Buffer (ones 2)))
    (define proc (lambda~> (+ weight) (+ offset)))
    (check-equal? (parameters (procedure->Layer proc)) '())
    (define net
      (procedure->Layer proc
                       #:parameters (list (cons "weight" weight))
                       #:buffers (list (cons "offset" offset))))
    (check-equal? (map car (named-parameters net)) '("weight"))
    (check-eq? (car (parameters net)) weight)
    (check-eq? (car (buffers net)) offset)
    (check-equal? (tensor->list (net (zeros 2))) '(2.0 2.0))
    (backward! (sum (net (zeros 2))))
    (check-true (has-grad? weight))
    (step! (sgd (parameters net) #:lr 0.5))
    (check-equal? (tensor->list weight) '(0.5 0.5))
    (check-equal? (tensor->list offset) '(1.0 1.0))
    (check-equal? (map car (state-dict net)) '("weight")))

  (test-case "registered children participate in training and containers"
    (define projection (Linear 2 2))
    (define drop (Dropout #:p 0.5))
    (define net
      (procedure->Layer (lambda~> projection relu drop)
                       #:children (list (cons "projection" projection)
                                        (cons "drop" drop))))
    (define parent (Sequential (list net)))
    (check-equal? (map car (named-parameters parent))
                  '("0.projection.weight" "0.projection.bias"))
    (check-equal? (length (parameters (LayerList (list net)))) 2)
    (eval! parent)
    (check-false (layer-training? drop))
    (check-equal? (tensor->list (parent (ones 2)))
                  (tensor->list (relu (projection (ones 2)))))
    (train! parent)
    (check-true (layer-training? drop)))

  (test-case "shared captured parameters are returned once"
    (define weight (Parameter (ones 2)))
    (define net
      (procedure->Layer values #:parameters (list (cons "a" weight)
                                                 (cons "b" weight))))
    (check-equal? (map car (named-parameters net)) '("a"))
    (check-equal? (length (parameters net)) 1))

  (test-case "invalid registrations blame the caller"
    (check-exn exn:fail:contract? (lambda () (procedure->Layer 1)))
    (check-exn exn:fail:contract?
               (lambda () (procedure->Layer values #:parameters
                                           (list (cons "weight" (ones 2))))))
    (check-exn exn:fail:contract?
               (lambda () (procedure->Layer values #:buffers
                                           (list (cons "buffer" (ones 2))))))
    (check-exn exn:fail:contract?
               (lambda () (procedure->Layer values #:children
                                           (list (cons "child" values)))))
    (check-exn exn:fail:contract?
               (lambda () (procedure->Layer values #:children
                                           (list (cons "a.b" (Linear 2 2))))))
    (check-exn exn:fail:contract?
               (lambda () (procedure->Layer values #:children
                                           (list (cons "a" (Dropout))
                                                 (cons "a" (Dropout))))))
    (check-exn exn:fail:contract?
               (lambda () (procedure->Layer values
                                           #:parameters (list (cons "a" (Parameter (ones 2))))
                                           #:buffers (list (cons "a" (Buffer (ones 2)))))))))

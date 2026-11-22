#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" * backward! ones sum to)
           (only-in "../nn.rkt"
                    Linear grad-scaler grad-scaler-scale-loss grad-scaler-state-dict
                    grad-scaler-step! grad-scaler-update! parameters sgd zero-grads!)
           (only-in "private/optim-oracle.rkt" call-with-optim-oracle oracle-json))

  (define device (string->symbol (or (getenv "RKTORCH_PARITY_DEVICE") "cpu")))

  (call-with-optim-oracle
   'grad-scaler-state-parity-test
   (lambda (dir)
     (test-case "T3-L31: grad scaler state after an overflow sequence equals torch's"
       (define net (to (Linear 3 2) device))
       (define opt (sgd (parameters net) #:lr 0.1))
       (define s (grad-scaler #:device device #:init-scale 16.0 #:growth-interval 2))
       (for ([torch (in-list (hash-ref (oracle-json dir "grad-scaler") 'states))] [t (in-naturals)])
         (define loss (sum (net (ones 1 3 #:device device))))
         (define scaled (grad-scaler-scale-loss s (if (hash-ref torch 'overflow) (* loss +inf.0) loss)))
         (backward! scaled)
         (grad-scaler-step! s opt)
         (grad-scaler-update! s)
         (zero-grads! opt)
         (check-equal? (grad-scaler-state-dict s)
                       (for/hash ([(k v) (in-hash (hash-remove torch 'overflow))])
                         (values (symbol->string k) v))
                       (format "step ~a" t)))))))

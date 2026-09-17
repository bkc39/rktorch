#lang racket/base

;; The parity cases need Python torch (inside `nix develop`) and self-skip
;; without it.

(module+ test
  (require (only-in racket/file make-temporary-file)
           (only-in racket/list append-map)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "../generated.rkt" cudnn-rnn-flatten-weight)
           "private/python-env.rkt")

  (define (flat ts)
    (append-map tensor->list ts))

  (define (check-close actual expected label [tolerance tol])
    (check-equal? (length actual) (length expected)
                  (format "~a: value count" label))
    (for ([a (in-list actual)] [e (in-list expected)] [i (in-naturals)])
      (check-= a e tolerance (format "~a: value ~a" label i))))

  (test-case "LSTM names its parameters as nn.LSTM does"
    (define net (LSTM 3 4 #:num-layers 2 #:bidirectional? #t))
    (check-equal? (map car (named-parameters net))
                  (for*/list ([layer '(0 1)]
                              [suffix '("" "_reverse")]
                              [stem '("weight_ih" "weight_hh"
                                      "bias_ih" "bias_hh")])
                    (format "~a_l~a~a" stem layer suffix)))
    (check-equal? (tensor-shape (cdr (assoc "weight_ih_l0" (named-parameters net))))
                  '(16 3))
    (check-equal? (tensor-shape
                   (cdr (assoc "weight_ih_l1_reverse" (named-parameters net))))
                  '(16 8))
    (check-equal? (length (parameters (GRU 3 4 #:bias? #f))) 2)
    (check-pred lstm? net)
    (check-false (gru? net))
    (check-pred layer? net))

  (test-case "forward answers output and final state as values"
    (define lstm (LSTM 3 4))
    (define-values (out h c) (lstm (randn 5 2 3)))
    (check-equal? (tensor-shape out) '(5 2 4))
    (check-equal? (tensor-shape h) '(1 2 4))
    (check-equal? (tensor-shape c) '(1 2 4))
    (define gru (GRU 3 4 #:batch-first? #t #:bidirectional? #t))
    (define-values (gru-out gru-h) (gru (randn 2 5 3)))
    (check-equal? (tensor-shape gru-out) '(2 5 8))
    (check-equal? (tensor-shape gru-h) '(2 2 4)))

  (test-case "a carried state continues the sequence"
    (manual-seed! 0)
    (define gru (GRU 3 4))
    (define x (randn 6 2 3))
    (define-values (whole _h) (gru x))
    (define-values (_first h-mid) (gru (narrow x 0 0 3)))
    (define-values (second _h-end) (gru (narrow x 0 3 3) h-mid))
    (check-close (tensor->list second) (tensor->list (narrow whole 0 3 3))
                 "second half" 1e-6))

  (test-case "the state arguments come all together or not at all"
    (define lstm (LSTM 3 4))
    (check-exn exn:fail:contract:arity?
               (lambda () (lstm (randn 5 2 3) (zeros 1 2 4))))
    (check-exn exn:fail:contract:arity?
               (lambda () ((GRU 3 4) (randn 5 2 3) (zeros 1 2 4) (zeros 1 2 4))))
    (check-exn #rx"LSTM: contract violation.*rank-3 tensor"
               (lambda () (lstm (randn 5 3))))
    (check-exn #rx"LSTM: contract violation.*rank-3 tensor"
               (lambda () (lstm '(1 2 3))))
    (check-exn #rx"GRU: contract violation.*rank-3 tensor"
               (lambda () ((GRU 3 4) (randn 5 2 3) (zeros 2 4)))))

  (test-case "inter-layer dropout is a training-mode behaviour"
    (manual-seed! 0)
    (define gru (GRU 3 4 #:num-layers 2 #:dropout 0.5))
    (define x (randn 5 2 3))
    (define (run)
      (define-values (out _h) (gru x))
      (tensor->list out))
    (check-not-equal? (run) (run))
    (eval! gru)
    (check-equal? (run) (run)))

  (test-case "to moves a recurrent layer through its own move path"
    (define gru (GRU 3 4))
    (check-eq? (to gru 'float64) gru)
    (check-equal? (map tensor-dtype (parameters gru))
                  '(float64 float64 float64 float64))
    (define-values (out _h) (gru (randn 5 2 3 #:dtype 'float64)))
    (check-equal? (tensor-dtype out) 'float64))

  (test-case "constructor contracts"
    (check-exn exn:fail:contract? (lambda () (LSTM 0 4)))
    (check-pred gru? (GRU 3 4 #:num-layers 2 #:dropout 1.0))
    (check-exn exn:fail:contract? (lambda () (GRU 3 4 #:dropout 1.5)))
    (check-exn exn:fail:contract? (lambda () (GRU 3 4 #:num-layers 0))))

  (test-case "a checkpoint round-trips through the PyTorch names"
    (define source (LSTM 3 4 #:bidirectional? #t))
    (define target (LSTM 3 4 #:bidirectional? #t))
    (define path (make-temporary-file "rkt-lstm-~a.safetensors"))
    (save-state! source path)
    (load-state! target path)
    (delete-file path)
    (check-equal? (map car (state-dict source))
                  (map car (named-parameters source)))
    (check-equal? (flat (parameters target)) (flat (parameters source))))

  (test-case "clip-grad-norm! scales every gradient by one factor"
    (define a (tensor '(3.0 0.0) #:requires-grad? #t))
    (define b (tensor '(0.0 4.0) #:requires-grad? #t))
    (define unused (tensor '(1.0) #:requires-grad? #t))
    (backward! (sum (+ (* a a 0.5) (* b b 0.5))))
    (define total (clip-grad-norm! (list a b unused) 1.0))
    (check-= (item total) 5.0 1e-6)
    (check-close (tensor->list (grad a)) '(0.6 0.0) "grad a" 1e-5)
    (check-close (tensor->list (grad b)) '(0.0 0.8) "grad b" 1e-5)
    (check-false (has-grad? unused)))

  (test-case "gradients already inside the bound are left alone"
    (define a (tensor '(0.3 0.4) #:requires-grad? #t))
    (backward! (sum (* a a 0.5)))
    (check-= (item (clip-grad-norm! (list a) 1.0)) 0.5 1e-6)
    (check-close (tensor->list (grad a)) '(0.3 0.4) "grad" 1e-6))

  (test-case "clip-grad-norm! with nothing to clip answers zero"
    (check-= (item (clip-grad-norm! '() 1.0)) 0.0 0.0)
    (check-exn exn:fail:contract? (lambda () (clip-grad-norm! '() -1))))

  (test-case "a bound of zero zeroes the gradients and still answers the norm"
    (define a (tensor '(3.0 4.0) #:requires-grad? #t))
    (backward! (sum (* a a 0.5)))
    (check-= (item (clip-grad-norm! (list a) 0)) 5.0 1e-6)
    (check-equal? (tensor->list (grad a)) '(0.0 0.0)))

  (define (check-layer-parity expected make label)
    (manual-seed! 0)
    (define net (make))
    (check-equal? (map car (named-parameters net)) (hash-ref expected 'names)
                  (format "~a: parameter names" label))
    (check-equal? (map tensor-shape (parameters net)) (hash-ref expected 'shapes)
                  (format "~a: parameter shapes" label))
    (check-close (flat (parameters net)) (hash-ref expected 'params)
                 (format "~a: seeded init" label))
    (define x (randn 2 5 3))
    (define results (call-with-values (lambda () (net x)) list))
    (check-equal? (tensor-shape (car results)) (hash-ref expected 'out_shape)
                  (format "~a: output shape" label))
    (check-close (tensor->list (car results)) (hash-ref expected 'out)
                 (format "~a: output" label) 1e-5)
    (for ([state (in-list (cdr results))]
          [py (in-list (hash-ref expected 'states))]
          [i (in-naturals)])
      (check-close (tensor->list state) py (format "~a: state ~a" label i) 1e-5))
    (backward! (for/fold ([acc (sum (car results))])
                         ([state (in-list (cdr results))])
                 (+ acc (sum state))))
    (define total (clip-grad-norm! (parameters net) 0.5))
    (check-= (item total) (hash-ref expected 'total_norm) 1e-3
             (format "~a: total norm" label))
    (check-close (flat (map grad (parameters net))) (hash-ref expected 'grads)
                 (format "~a: clipped gradients" label) 1e-5))

  (cond
    [(not (python-torch-available?))
     (displayln "[recurrent-layer-test] parity skipped: python3 `torch` not available")]
    [else
     (define j (python-check "recurrent_layers.py"))
     (check-layer-parity (hash-ref j 'lstm)
                         (lambda () (LSTM 3 4 #:num-layers 2 #:bidirectional? #t
                                          #:batch-first? #t))
                         "LSTM")
     (check-layer-parity (hash-ref j 'gru)
                         (lambda () (GRU 3 4 #:num-layers 2 #:bidirectional? #t
                                         #:batch-first? #t))
                         "GRU")
     (check-layer-parity (hash-ref j 'gru_plain)
                         (lambda () (GRU 3 4 #:bias? #f #:batch-first? #t))
                         "GRU without bias")])

  (when (cuda-available?)
    (test-case "cudnn-rnn-flatten-weight packs the weights into one buffer"
      (manual-seed! 0)
      (define gru (to (GRU 3 4 #:num-layers 2 #:bidirectional? #t) 'cuda))
      (define weights (parameters gru))
      (define before (flat (map (lambda (w) (to-device w 'cpu)) weights)))
      (define buffer
        (with-no-grad
          (cudnn-rnn-flatten-weight weights 4 3 3 4 0 2 #f #t)))
      (check-equal? (device-type (tensor-device buffer)) 'cuda)
      (check-true (>= (tensor-numel buffer)
                      (for/sum ([w (in-list weights)]) (tensor-numel w))))
      (check-equal? (flat (map (lambda (w) (to-device w 'cpu)) weights)) before)
      (check-exn #rx"cudnn-rnn-flatten-weight"
                 (lambda ()
                   (with-no-grad
                     (cudnn-rnn-flatten-weight (list (car weights))
                                               4 3 3 4 0 2 #f #t)))))

    (test-case "a layer moved to CUDA flattens its weights and still agrees"
      (manual-seed! 0)
      (define lstm (LSTM 3 4 #:num-layers 2))
      (define x (randn 5 2 3))
      (define-values (cpu-out _h _c) (lstm x))
      (define before (flat (parameters lstm)))
      (to lstm 'cuda)
      (define-values (gpu-out _gh _gc) (lstm (to-device x 'cuda)))
      (check-close (tensor->list (to-device gpu-out 'cpu)) (tensor->list cpu-out)
                   "cudnn output" 1e-4)
      (check-close (flat (for/list ([p (in-list (parameters lstm))])
                           (to-device p 'cpu)))
                   before "weights survive flattening" 0.0)
      (backward! (sum gpu-out))
      (for ([p (in-list (parameters lstm))])
        (check-equal? (tensor-shape (grad p)) (tensor-shape p)))
      (to lstm 'cpu)
      (to lstm 'cuda)
      (define-values (round-trip _rh _rc) (lstm (to-device x 'cuda)))
      (check-close (tensor->list (to-device round-trip 'cpu))
                   (tensor->list cpu-out) "after a CPU round trip" 1e-4)
      (define opt (adam (parameters lstm)))
      (step! opt)
      (define-values (after-step _ah _ac) (lstm (to-device x 'cuda)))
      (check-equal? (tensor-shape after-step) '(5 2 4)))))

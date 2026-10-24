#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../foreign.rkt"
                    accelerator-if-available copy! device-type zeros-like)
           (only-in "../foreign/ops.rkt" move-generation)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    backward! grad manual-seed! ones randn sum
                    tensor->list to)
           (only-in "../nn.rkt"
                    Linear Sequential adam parameters rmsprop sgd step!
                    zero-grads!)
           (only-in (submod "../nn/optim.rkt" private) optimizer-state)
           (only-in "../scribblings/examples/05-mnist.rkt" convnet)
           (only-in "../vision/resnet.rkt" ResNet)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/reference-optim.rkt"
                    reference-adam reference-rmsprop reference-sgd)
           (only-in "private/within.rkt" bounded-test))

  (define (copy-of p)
    (define q (zeros-like p))
    (copy! q p)
    q)

  (define (values-of ts)
    (map tensor->list ts))

  (define ((placed-on target) t)
    (if target (to t target) t))

  (define (twin-steps make-net input make-opt reference
                      #:target [target #f]
                      #:move [move (lambda (net _opt) ((placed-on target) net))])
    (manual-seed! 0)
    (define net (make-net))
    (define ps (parameters net))
    (define opt (make-opt ps))
    (define (step-and-check qs x n)
      (backward! (sum (net x)))
      (define gs (map grad ps))
      (step! opt)
      (reference qs gs)
      (zero-grads! opt)
      (check-equal? (values-of ps) (values-of qs) (format "step ~a" n)))
    (define qs (map copy-of ps))
    (step-and-check qs input 1)
    (move net opt)
    (define placed (placed-on target))
    (define moved (map placed qs))
    (define x (placed input))
    (for ([n (in-list '(2 3))])
      (step-and-check moved x n))
    opt)

  (define optimizers
    (list (list "sgd" 5
                (lambda (ps) (sgd ps #:lr 0.1))
                (lambda () (reference-sgd #:lr 0.1)))
          (list "sgd, Nesterov momentum and weight decay" 19
                (lambda (ps)
                  (sgd ps #:lr 0.1 #:momentum 0.9 #:nesterov? #t
                       #:weight-decay 5e-4))
                (lambda ()
                  (reference-sgd #:lr 0.1 #:momentum 0.9 #:nesterov? #t
                                 #:weight-decay 5e-4)))
          (list "adam" 14
                (lambda (ps) (adam ps))
                (lambda () (reference-adam)))
          (list "adam, weight decay" 20
                (lambda (ps) (adam ps #:lr 0.01 #:weight-decay 1e-2))
                (lambda () (reference-adam #:lr 0.01 #:weight-decay 1e-2)))
          (list "rmsprop" 11
                (lambda (ps) (rmsprop ps #:lr 0.01))
                (lambda () (reference-rmsprop #:lr 0.01)))
          (list "rmsprop, momentum and weight decay" 19
                (lambda (ps)
                  (rmsprop ps #:lr 0.01 #:momentum 0.5 #:weight-decay 1e-3))
                (lambda ()
                  (reference-rmsprop #:lr 0.01 #:momentum 0.5
                                     #:weight-decay 1e-3)))))

  (define (mlp)
    (Sequential (Linear 4 8) (Linear 8 3)))

  (define models
    (list (list "05's convnet" convnet '(2 1 28 28))
          (list "09's ResNet" (lambda () (ResNet #:base 4)) '(2 3 32 32))))

  (for* ([model (in-list models)]
         [o (in-list optimizers)])
    (match-define (list model-name make-net dims) model)
    (match-define (list name _ make-opt make-reference) o)
    (bounded-test
     (format "~a: three ~a steps match the reference's, bit for bit"
             model-name name)
     (twin-steps make-net (apply randn dims) make-opt (make-reference))))

  (define (step-in-the-last-move! net opt target)
    (define here (current-thread))
    (define last-move (+ (move-generation) (length (parameters net))))
    (define fired? #f)
    (with-crossing-hook (lambda ()
                          (when (and (eq? (current-thread) here)
                                     (not fired?)
                                     (= (move-generation) last-move))
                            (set! fired? #t)
                            (step! opt)))
      (to net target)))

  (define accelerator (accelerator-if-available))

  (for ([o (in-list optimizers)])
    (match-define (list name _ make-opt make-reference) o)
    (bounded-test
     (format "~a: a dtype change after the first step moves the state along"
             name)
     (twin-steps mlp (randn 2 4) make-opt (make-reference) #:target 'float64))
    (bounded-test
     (format "~a: a step during a layer's move still follows it" name)
     (twin-steps (lambda () (Linear 4 3)) (randn 2 4) make-opt
                 (make-reference)
                 #:target 'float64
                 #:move (lambda (net opt)
                          (step-in-the-last-move! net opt 'float64))))
    (bounded-test
     (format "~a: another layer's move keeps this optimizer's state" name)
     (define before #f)
     (define opt
       (twin-steps mlp (randn 2 4) make-opt (make-reference)
                   #:move (lambda (_net optimizer)
                            (set! before (optimizer-state optimizer))
                            (to (Linear 2 2) 'float64))))
     (define after (optimizer-state opt))
     (check-equal? (length after) (length before))
     (check-true (andmap (lambda (s) (and (memq s after) #t)) before)))
    (bounded-test
     (format "~a: a move to ~a after the first step moves the state along"
             name (device-type accelerator))
     (twin-steps mlp (randn 2 4) make-opt (make-reference)
                 #:target accelerator)))

  (define (op-calls-of thunk)
    (define here (current-thread))
    (define crossings 0)
    (define frees 0)
    (define tensors 0)
    (parameterize ([native-collect-at-troughs #f])
      (with-crossing-hook (lambda ()
                            (when (eq? (current-thread) here)
                              (set! crossings (add1 crossings))))
        (with-race-hook (lambda (name _subject)
                          (case (and (eq? (current-thread) here) name)
                            [(ledger-entry-added) (set! tensors (add1 tensors))]
                            [(free-released finalizer-releasing)
                             (set! frees (add1 frees))]
                            [else (void)]))
          (thunk))))
    (- crossings frees (* 2 tensors)))

  (define (step-op-calls net opt target)
    (backward! (sum (net ((placed-on target) (ones 2 3)))))
    (begin0 (op-calls-of (lambda () (step! opt)))
      (zero-grads! opt)))

  (define (per-parameter make-opt)
    (define (costs depth)
      (define net
        (apply Sequential (for/list ([_ (in-range depth)]) (Linear 3 3))))
      (define opt (make-opt (parameters net)))
      (define steady
        (for/last ([_ (in-range 3)])
          (step-op-calls net opt #f)))
      (to net 'float64)
      (define moving (step-op-calls net opt 'float64))
      (list steady moving (step-op-calls net opt 'float64)))
    (map (lambda (one two) (/ (- two one) 2)) (costs 1) (costs 2)))

  (for ([o (in-list optimizers)])
    (match-define (list name bound make-opt _) o)
    (bounded-test
     (format "a steady ~a step makes at most ~a op calls a parameter"
             name bound)
     (match-define (list steady moving after) (per-parameter make-opt))
     (check <= steady bound "steady")
     (check >= moving steady "the step after a move")
     (check-equal? after steady "the step after that"))))

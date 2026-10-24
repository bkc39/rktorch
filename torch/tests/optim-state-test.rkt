#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../foreign.rkt"
                    accelerator-if-available copy! device-type zeros-like)
           (only-in (submod "../foreign.rkt" unsafe) to!)
           (only-in "../foreign/ops.rkt" move-generation)
           (only-in "../foreign/raw/autograd.rkt" tr-set-grad-enabled/raw)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    backward! dtype grad manual-seed! ones randn sum
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
    (if target (apply to t target) t))

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

  (define-runtime-path optim-file "../nn/optim.rkt")
  (define-runtime-path foreign-file "../foreign.rkt")

  (define (fresh-optimizer name)
    (define from (variable-reference->namespace (#%variable-reference)))
    (define ns (make-base-empty-namespace))
    (for ([m (in-list (list foreign-file
                            'racket/contract/base
                            'racket/generic))])
      (namespace-attach-module from m ns))
    (parameterize ([current-namespace ns])
      (values (dynamic-require optim-file name)
              (dynamic-require optim-file 'optimizer-step!))))

  (struct injected (crossing) #:transparent)

  (define (stepped-linear)
    (define net (Linear 4 3))
    (backward! (sum (net (randn 2 4))))
    net)

  (define (faulted-at? k make-opt step)
    (define opt (make-opt (parameters (stepped-linear))))
    (define here (current-thread))
    (define crossings 0)
    (with-handlers ([injected? (lambda (_) #t)])
      (with-crossing-hook (lambda ()
                            (when (eq? (current-thread) here)
                              (set! crossings (add1 crossings))
                              (when (= crossings k)
                                (raise (injected k)))))
        (step opt))
      #f))

  (define (crossings-survived name)
    (let loop ([k 1])
      (define-values (make-opt step) (fresh-optimizer name))
      (define faulted? (faulted-at? k make-opt step))
      (void (tr-set-grad-enabled/raw #t))
      (step (make-opt (parameters (stepped-linear))))
      (if faulted? (loop (add1 k)) (sub1 k))))

  (for ([name (in-list '(adam rmsprop))])
    (bounded-test
     (format "a fault at any crossing of a first ~a step spares later ones"
             name)
     (check-true (positive? (crossings-survived name)))))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (with-crossing-hook (lambda ()
                          (when (eq? (current-thread) here)
                            (set! n (add1 n))))
      (thunk))
    n)

  (bounded-test "a break at a move's native crossing waits for its generation"
    (define placed (to (randn 2 2) 'float64))
    (define reads (crossings-of (lambda () (to! placed 'float64))))
    (define t (randn 2 2))
    (define before (move-generation))
    (define here (current-thread))
    (define n 0)
    (define broke?
      (with-handlers ([exn:break? (lambda (_) #t)])
        (with-crossing-hook (lambda ()
                              (when (eq? (current-thread) here)
                                (set! n (add1 n))
                                (when (= n (add1 reads))
                                  (break-thread here))))
          (to! t 'float64))
        #f))
    (check-true broke?)
    (check-eq? (dtype t) 'float64)
    (check-equal? (move-generation) (add1 before)))

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
      (apply to net target))
    fired?)

  (define accelerator (accelerator-if-available))

  (define accelerator-target
    (if (eq? (device-type accelerator) 'cpu)
        (list accelerator 'float64)
        (list accelerator)))

  (for ([o (in-list optimizers)])
    (match-define (list name _ make-opt make-reference) o)
    (bounded-test
     (format "~a: a dtype change after the first step moves the state along"
             name)
     (twin-steps mlp (randn 2 4) make-opt (make-reference) #:target '(float64)))
    (bounded-test
     (format "~a: a step during a layer's move still follows it" name)
     (twin-steps (lambda () (Linear 4 3)) (randn 2 4) make-opt
                 (make-reference)
                 #:target '(float64)
                 #:move (lambda (net opt)
                          (check-true
                           (step-in-the-last-move! net opt '(float64))))))
    (bounded-test
     (format "~a: another layer's move leaves this optimizer's steps alone"
             name)
     (twin-steps mlp (randn 2 4) make-opt (make-reference)
                 #:move (lambda (_net _opt) (to (Linear 2 2) 'float64))))
    (bounded-test
     (format "~a: a move to ~a after the first step moves the state along"
             name (map (lambda (x) (if (symbol? x) x (device-type x)))
                       accelerator-target))
     (twin-steps mlp (randn 2 4) make-opt (make-reference)
                 #:target accelerator-target)))

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
      (define moving (step-op-calls net opt '(float64)))
      (define after (step-op-calls net opt '(float64)))
      (to (Linear 2 2) 'float64)
      (define unrelated (step-op-calls net opt '(float64)))
      (list steady moving after unrelated (length (optimizer-state opt))))
    (map (lambda (one two) (/ (- two one) 2)) (costs 1) (costs 2)))

  (define re-check-reads 4)

  (for ([o (in-list optimizers)])
    (match-define (list name bound make-opt _) o)
    (bounded-test
     (format "a steady ~a step makes at most ~a op calls a parameter"
             name bound)
     (match-define (list steady moving after unrelated held)
       (per-parameter make-opt))
     (check <= steady bound "steady")
     (check (if (zero? held) = >) moving steady "the step after a move")
     (check-equal? after steady "the step after that")
     (check <= unrelated (+ steady (* re-check-reads held))
            "the step after another layer's move"))))

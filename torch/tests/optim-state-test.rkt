#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../foreign.rkt" copy! cuda-available? zeros-like)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../main.rkt"
                    backward! grad manual-seed! ones randn sum
                    tensor->list to)
           (only-in "../nn.rkt"
                    Linear Sequential adam parameters rmsprop sgd step!
                    zero-grads!)
           (only-in "../scribblings/examples/05-mnist.rkt" convnet)
           (only-in "../vision/resnet.rkt" ResNet)
           (only-in "private/reference-optim.rkt"
                    reference-adam reference-rmsprop reference-sgd)
           (only-in "private/within.rkt" bounded-test within))

  (define (bounded name thunk)
    (test-case name (within 60 thunk)))

  (define (copy-of p)
    (define q (zeros-like p))
    (copy! q p)
    q)

  (define (values-of ts)
    (map tensor->list ts))

  (define (twin-steps make-net input make-opt reference
                      #:steps [steps 3] #:move-to [target #f] #:between [between void])
    (manual-seed! 0)
    (define net (make-net))
    (define ps (parameters net))
    (define opt (make-opt ps))
    (for/fold ([qs (map copy-of ps)] [x input] #:result (void))
              ([i (in-range steps)])
      (define move? (and target (= i 1)))
      (when (= i 1)
        (between))
      (when move?
        (to net target))
      (define qs* (if move? (for/list ([q (in-list qs)]) (to q target)) qs))
      (define x* (if move? (to x target) x))
      (backward! (sum (net x*)))
      (define gs (map grad ps))
      (step! opt)
      (reference qs* gs)
      (zero-grads! opt)
      (check-equal? (values-of ps) (values-of qs*) (format "step ~a" (add1 i)))
      (values qs* x*)))

  (define optimizers
    (list (list "sgd" (lambda (ps) (sgd ps #:lr 0.1)) (lambda () (reference-sgd #:lr 0.1)))
          (list "sgd, Nesterov momentum and weight decay"
                (lambda (ps) (sgd ps #:lr 0.1 #:momentum 0.9 #:nesterov? #t #:weight-decay 5e-4))
                (lambda () (reference-sgd #:lr 0.1 #:momentum 0.9 #:nesterov? #t
                                          #:weight-decay 5e-4)))
          (list "adam" (lambda (ps) (adam ps)) (lambda () (reference-adam)))
          (list "adam, weight decay"
                (lambda (ps) (adam ps #:lr 0.01 #:weight-decay 1e-2))
                (lambda () (reference-adam #:lr 0.01 #:weight-decay 1e-2)))
          (list "rmsprop" (lambda (ps) (rmsprop ps #:lr 0.01))
                (lambda () (reference-rmsprop #:lr 0.01)))
          (list "rmsprop, momentum and weight decay"
                (lambda (ps) (rmsprop ps #:lr 0.01 #:momentum 0.5 #:weight-decay 1e-3))
                (lambda () (reference-rmsprop #:lr 0.01 #:momentum 0.5 #:weight-decay 1e-3)))))

  (define (mlp)
    (Sequential (Linear 4 8) (Linear 8 3)))

  (define models
    (list (list "05's convnet" convnet '(2 1 28 28))
          (list "09's ResNet" (lambda () (ResNet #:base 4)) '(2 3 32 32))))

  (for* ([model (in-list models)]
         [o (in-list optimizers)])
    (match-define (list model-name make-net dims) model)
    (match-define (list name make-opt make-reference) o)
    (bounded (format "~a: three ~a steps match e179ba7's, bit for bit" model-name name)
      (lambda () (twin-steps make-net (apply randn dims) make-opt (make-reference)))))

  (for ([o (in-list optimizers)])
    (match-define (list name make-opt make-reference) o)
    (bounded (format "~a: a dtype change after the first step moves the state along" name)
      (lambda () (twin-steps mlp (randn 2 4) make-opt (make-reference) #:move-to 'float64)))
    (bounded (format "~a: another layer's move leaves the state where it is" name)
      (lambda ()
        (twin-steps mlp (randn 2 4) make-opt (make-reference)
                    #:between (lambda () (to (Linear 2 2) 'float64)))))
    (bounded (format "~a: a move to CUDA after the first step moves the state along" name)
      (lambda ()
        (cond
          [(cuda-available?)
           (twin-steps mlp (randn 2 4) make-opt (make-reference) #:move-to 'cuda)]
          [else (displayln "skipped: no CUDA device")]))))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (with-crossing-hook (lambda () (when (eq? (current-thread) here) (set! n (add1 n))))
      (thunk))
    n)

  (define (step-crossings net opt)
    (backward! (sum (net (ones 2 3))))
    (begin0 (crossings-of (lambda () (step! opt)))
      (zero-grads! opt)))

  (define (per-parameter make-opt)
    (define (steady depth)
      (define net (apply Sequential (for/list ([_ (in-range depth)]) (Linear 3 3))))
      (define opt (make-opt (parameters net)))
      (for/last ([_ (in-range 3)])
        (step-crossings net opt)))
    (/ (- (steady 2) (steady 1)) 2))

  (for ([o (in-list optimizers)]
        [bound (in-list '(8 30 19 29 16 28))])
    (match-define (list name make-opt _) o)
    (bounded (format "a steady ~a step costs at most ~a native calls a parameter" name bound)
      (lambda () (check <= (per-parameter make-opt) bound))))

  (bounded-test "a move costs the next step its re-placement, and only that step"
    (define net (Sequential (Linear 3 3)))
    (define opt (adam (parameters net) #:lr 0.1))
    (define before (for/last ([_ (in-range 3)]) (step-crossings net opt)))
    (to net 'float64)
    (define (step-64)
      (backward! (sum (net (to (ones 2 3) 'float64))))
      (begin0 (crossings-of (lambda () (step! opt)))
        (zero-grads! opt)))
    (check > (step-64) before "the step after the move re-places the moments")
    (check-equal? (step-64) before "the step after that is back to the floor")))

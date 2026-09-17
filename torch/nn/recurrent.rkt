#lang racket/base

(require (only-in racket/contract/base
                  -> ->* </c >=/c and/c any/c contract-out)
         (only-in "../foreign.rkt"
                  device-type prop:to tensor-device tensor-dtype tensor-shape
                  tensor? with-no-grad zeros)
         (only-in "../generated.rkt"
                  cudnn-rnn-flatten-weight gru-input lstm-input)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "init.rkt" uniform-init)
         (only-in "layer.rkt" gen:layer move-layer! training?)
         (only-in "parameter.rkt" Parameter))

(provide (contract-out [lstm? (-> any/c boolean?)]
                       [gru? (-> any/c boolean?)]))

(struct recurrent (input-size hidden-size num-layers bias? batch-first?
                   dropout bidirectional? params
                   [mode #:mutable] [flattened-on #:mutable])
  #:property prop:procedure
  (lambda (self x . state) (run self x state))
  #:methods gen:layer
  [(define (layer-forward self . inputs)
     (run self (car inputs) (cdr inputs)))
   (define (layer-parameters self)
     (map cdr (recurrent-params self)))
   (define (layer-named-parameters self prefix)
     (for/list ([p (in-list (recurrent-params self))])
       (cons (string-append prefix (car p)) (cdr p))))
   (define (layer-mode self)
     (recurrent-mode self))
   (define (layer-set-mode! self mode)
     (set-recurrent-mode! self mode))])

;; A move rebinds every parameter to scattered storage, on any device and
;; for a dtype change alike, so it is the move that forgets the flattening.
(define (move-and-scatter! self device dtype)
  (set-recurrent-flattened-on! self #f)
  (move-layer! self device dtype))

(struct lstm recurrent ()
  #:reflection-name 'LSTM
  #:property prop:to move-and-scatter!)

(struct gru recurrent ()
  #:reflection-name 'GRU
  #:property prop:to move-and-scatter!)

;; cudnnRNNMode_t
(define (cudnn-mode self)
  (if (lstm? self) 2 3))

(define (direction-count self)
  (if (recurrent-bidirectional? self) 2 1))

;; nn.RNNBase's order, which is both the flat-weight order ATen expects and
;; reset_parameters' draw order: per layer, per direction, w_ih w_hh b_ih b_hh
(define (draw-parameters gates input-size hidden-size num-layers bias?
                         bidirectional?)
  (define rows (* gates hidden-size))
  (define bound (/ 1.0 (sqrt hidden-size)))
  (define (draw . dims)
    (Parameter (uniform-init dims (- bound) bound)))
  (for*/list ([layer (in-range num-layers)]
              [suffix (in-list (if bidirectional? '("" "_reverse") '("")))]
              [entry
               (in-list
                (let ([fan-in (if (zero? layer)
                                  input-size
                                  (* hidden-size (if bidirectional? 2 1)))])
                  (append
                   (list (list "weight_ih" rows fan-in)
                         (list "weight_hh" rows hidden-size))
                   (if bias?
                       (list (list "bias_ih" rows)
                             (list "bias_hh" rows))
                       '()))))])
    (cons (format "~a_l~a~a" (car entry) layer suffix)
          (apply draw (cdr entry)))))

(define dropout/c (and/c real? (>=/c 0) (</c 1)))

(define-syntax-rule (define-recurrent-layer Name make made? gates)
  (define/contract-out (Name input-size hidden-size ;; noqa
                             #:num-layers [num-layers 1]
                             #:bias? [bias? #t]
                             #:batch-first? [batch-first? #f]
                             #:dropout [dropout 0.0]
                             #:bidirectional? [bidirectional? #f])
    (->* [exact-positive-integer? exact-positive-integer?]
         [#:num-layers exact-positive-integer?
          #:bias? boolean?
          #:batch-first? boolean?
          #:dropout dropout/c
          #:bidirectional? boolean?]
         made?)
    (make input-size hidden-size num-layers bias? batch-first?
          (exact->inexact dropout) bidirectional?
          (draw-parameters gates input-size hidden-size num-layers bias?
                           bidirectional?)
          'train #f)))

(define-recurrent-layer LSTM lstm lstm? 4)
(define-recurrent-layer GRU gru gru? 3)

(define (weights self)
  (map cdr (recurrent-params self)))

;; Flattening is an optimisation cudnn asks for, never a requirement: a
;; build or a dtype it refuses still runs, on compacted copies.
(define (flatten-weights! self)
  (with-handlers ([exn:fail? void])
    (with-no-grad
      (void
       (cudnn-rnn-flatten-weight (weights self)
                                 (if (recurrent-bias? self) 4 2)
                                 (recurrent-input-size self)
                                 (cudnn-mode self)
                                 (recurrent-hidden-size self)
                                 0
                                 (recurrent-num-layers self)
                                 (recurrent-batch-first? self)
                                 (recurrent-bidirectional? self))))))

(define (ensure-flat! self)
  (define device (tensor-device (car (weights self))))
  (unless (equal? device (recurrent-flattened-on self))
    (when (eq? (device-type device) 'cuda)
      (flatten-weights! self))
    (set-recurrent-flattened-on! self device)))

(define (zero-state self x)
  (define batch
    (list-ref (tensor-shape x) (if (recurrent-batch-first? self) 0 1)))
  (zeros (* (recurrent-num-layers self) (direction-count self))
         batch
         (recurrent-hidden-size self)
         #:device (tensor-device x)
         #:dtype (tensor-dtype x)))

(define (run self x state)
  (define who (if (lstm? self) 'LSTM 'GRU))
  (define state-count (if (lstm? self) 2 1))
  (unless (and (tensor? x) (= 3 (length (tensor-shape x))))
    (raise-argument-error who "a rank-3 tensor?" x))
  (unless (memv (length state) (list 0 state-count))
    (apply raise-arity-error who (list 1 (add1 state-count)) x state))
  (for ([s (in-list state)] [i (in-naturals 1)])
    (unless (and (tensor? s) (= 3 (length (tensor-shape s))))
      (apply raise-argument-error who "a rank-3 tensor?" i x state)))
  (ensure-flat! self)
  (define initial
    (if (null? state)
        (for/list ([_ (in-range state-count)]) (zero-state self x))
        state))
  (define (recur op hx)
    (op x hx (weights self)
        (recurrent-bias? self)
        (recurrent-num-layers self)
        (recurrent-dropout self)
        (training? (recurrent-mode self))
        (recurrent-bidirectional? self)
        (recurrent-batch-first? self)))
  (if (lstm? self)
      (recur lstm-input initial)
      (recur gru-input (car initial))))

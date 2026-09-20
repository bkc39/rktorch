#lang racket/base

(require (only-in racket/contract/base
                  -> ->* <=/c >=/c and/c contract-out)
         (only-in racket/match match)
         (only-in "../foreign.rkt"
                  device-type exn:fail:rktorch:oom? tensor-device tensor-dtype
                  tensor-shape tensor? with-no-grad zeros)
         (only-in "../generated.rkt"
                  cudnn-rnn-flatten-weight gru-input lstm-input)
         (only-in "init.rkt" uniform-init)
         (only-in "layer.rkt"
                  define-layer parameters-by-key training? with-mode)
         (only-in "parameter.rkt" Parameter))

(struct rnn (who gates state-count cudnn-mode input-size hidden-size
             num-layers bias? batch-first? dropout bidirectional?))

;; nn.RNNBase's order, which is both the flat-weight order ATen expects and
;; reset_parameters' draw order: per layer, per direction, w_ih w_hh b_ih b_hh
(define (draw-parameters spec)
  (define rows (* (rnn-gates spec) (rnn-hidden-size spec)))
  (define hidden (rnn-hidden-size spec))
  (define bound (/ 1.0 (sqrt hidden)))
  (define (draw . dims)
    (Parameter (uniform-init dims (- bound) bound)))
  (for*/list ([layer (in-range (rnn-num-layers spec))]
              [suffix (in-list (if (rnn-bidirectional? spec)
                                   '("" "_reverse")
                                   '("")))]
              [entry
               (in-list
                (let ([fan-in (if (zero? layer)
                                  (rnn-input-size spec)
                                  (* hidden (if (rnn-bidirectional? spec) 2 1)))])
                  (append
                   (list (list "weight_ih" rows fan-in)
                         (list "weight_hh" rows hidden))
                   (if (rnn-bias? spec)
                       (list (list "bias_ih" rows)
                             (list "bias_hh" rows))
                       '()))))])
    (cons (format "~a_l~a~a" (car entry) layer suffix)
          (apply draw (cdr entry)))))

;; Keyed on the first weight, which `to!` mutates in place and so survives a
;; move. A move that rebinds the storage drops the entry through `#:on-move`,
;; which a placement read at forward time could not do: a device round trip
;; ends where it started.
(define flattened (make-weak-hasheq))

;; Which weights cudnn refused to flatten. The refusal is swallowed so the
;; layer still runs, and the placement is cached either way so it is not asked
;; twice, which together would hide a defect in the arguments we pass as a
;; silent fall back to compacted copies; recording it lets a test say the
;; flattening happened rather than only that the outputs agree.
(define refused (make-weak-hasheq))

(define (placement weights)
  (define w (car weights))
  (cons (tensor-device w) (tensor-dtype w)))

;; Flattening is an optimisation cudnn asks for, never a requirement: a build
;; or a dtype it refuses still runs, on compacted copies, and refusing again
;; on the next call would cost an FFI round trip for the same answer. Two
;; failures are not that and reach the caller: an OOM, which is transient, so
;; the layer stays unflattened and retries once the pressure clears; and a
;; contract violation, which is a defect here rather than an answer from
;; cudnn, and would otherwise read as the fallback path.
(define (flatten-weights! spec weights)
  (with-handlers ([exn:fail:rktorch:oom? raise]
                  [exn:fail:contract? raise]
                  [exn:fail? (lambda (_e) (hash-set! refused (car weights) #t))])
    (with-no-grad
      (void
       (cudnn-rnn-flatten-weight weights
                                 (if (rnn-bias? spec) 4 2)
                                 (rnn-input-size spec)
                                 (rnn-cudnn-mode spec)
                                 (rnn-hidden-size spec)
                                 0
                                 (rnn-num-layers spec)
                                 (rnn-batch-first? spec)
                                 (rnn-bidirectional? spec))))))

(define (ensure-flat! spec weights)
  (define now (placement weights))
  (unless (equal? now (hash-ref flattened (car weights) #f))
    (when (eq? (device-type (car now)) 'cuda)
      (hash-remove! refused (car weights))
      (flatten-weights! spec weights))
    (hash-set! flattened (car weights) now)))

(define (forget-flattening! entries)
  (hash-remove! flattened (cdr (car entries))))

(define (zero-state spec x)
  (define batch
    (match (tensor-shape x)
      [(list batch-first-dim time-first-dim _)
       (if (rnn-batch-first? spec) batch-first-dim time-first-dim)]))
  (zeros (* (rnn-num-layers spec) (if (rnn-bidirectional? spec) 2 1))
         batch
         (rnn-hidden-size spec)
         #:device (tensor-device x)
         #:dtype (tensor-dtype x)))

;; raise-arity-error prints no expected line for an arity that is neither one
;; number nor a lower bound, and these accept the input alone or the input and
;; a full state, so the message is built here.
(define (raise-state-arity who count state)
  (raise (exn:fail:contract:arity
          (format (string-append "~a: arity mismatch;\n"
                                 " the expected number of arguments does not"
                                 " match the given number\n"
                                 "  expected: 1 or ~a\n"
                                 "  given: ~a")
                  who (add1 count) (add1 (length state)))
          (current-continuation-marks))))

(define (check-inputs spec x state)
  (define who (rnn-who spec))
  (unless (and (tensor? x) (= 3 (length (tensor-shape x))))
    (raise-argument-error who "a rank-3 tensor?" x))
  (unless (memv (length state) (list 0 (rnn-state-count spec)))
    (raise-state-arity who (rnn-state-count spec) state))
  (for ([s (in-list state)] [i (in-naturals 1)])
    (unless (and (tensor? s) (= 3 (length (tensor-shape s))))
      (apply raise-argument-error who "a rank-3 tensor?" i x state))))

(define (run spec entries op x state mode)
  (check-inputs spec x state)
  (define weights (map cdr entries))
  (ensure-flat! spec weights)
  (define initial
    (if (null? state)
        (for/list ([_ (in-range (rnn-state-count spec))]) (zero-state spec x))
        state))
  (op x
      (if (= 1 (rnn-state-count spec)) (car initial) initial)
      weights
      (rnn-bias? spec)
      (rnn-num-layers spec)
      (rnn-dropout spec)
      (training? mode)
      (rnn-bidirectional? spec)
      (rnn-batch-first? spec)))

(define dropout/c (and/c real? (>=/c 0) (<=/c 1)))

(define-layer LSTM (spec entries params) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:num-layers exact-positive-integer?
                   #:bias? boolean?
                   #:batch-first? boolean?
                   #:dropout dropout/c
                   #:bidirectional? boolean?]
                  lstm?)
  #:init (input-size hidden-size
          #:num-layers [num-layers 1]
          #:bias? [bias? #t]
          #:batch-first? [batch-first? #f]
          #:dropout [dropout 0.0]
          #:bidirectional? [bidirectional? #f])
  (set! spec (rnn 'LSTM 4 2 2 input-size hidden-size num-layers bias?
                  batch-first? (exact->inexact dropout) bidirectional?))
  (set! entries (draw-parameters spec))
  (set! params (parameters-by-key entries))
  #:on-move (forget-flattening! entries)
  #:forward (x . state)
  (with-mode (run spec entries lstm-input x state mode)))

(define-layer GRU (spec entries params) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:num-layers exact-positive-integer?
                   #:bias? boolean?
                   #:batch-first? boolean?
                   #:dropout dropout/c
                   #:bidirectional? boolean?]
                  gru?)
  #:init (input-size hidden-size
          #:num-layers [num-layers 1]
          #:bias? [bias? #t]
          #:batch-first? [batch-first? #f]
          #:dropout [dropout 0.0]
          #:bidirectional? [bidirectional? #f])
  (set! spec (rnn 'GRU 3 1 3 input-size hidden-size num-layers bias?
                  batch-first? (exact->inexact dropout) bidirectional?))
  (set! entries (draw-parameters spec))
  (set! params (parameters-by-key entries))
  #:on-move (forget-flattening! entries)
  #:forward (x . state)
  (with-mode (run spec entries gru-input x state mode)))

(module+ private
  (provide flattened-placement flatten-refused?)
  (define (flattened-placement weight) (hash-ref flattened weight #f))
  (define (flatten-refused? weight) (hash-ref refused weight #f)))

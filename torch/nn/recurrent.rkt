#lang racket/base

(require (only-in racket/contract/base
                  -> ->* <=/c >=/c and/c contract-out)
         (only-in racket/match == match match-define)
         (only-in "../foreign.rkt"
                  device device-type dtype exn:fail:rktorch:oom? tensor-shape
                  tensor? with-no-grad zeros)
         (only-in "../foreign/addresses.rkt"
                  tensor-data-ptr tensor-storage-ptr)
         (only-in "../foreign/raw/cas.rkt" cas-add!)
         (only-in "../foreign/raw/fault.rkt" native-fault?)
         (only-in "../generated.rkt"
                  cudnn-rnn-flatten-weight gru-input lstm-input)
         (only-in "../private/lock.rkt" make-lock with-lock)
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

;; packed weights are views of one storage, so compare the storage address too
(define flattened (make-weak-hasheq))

(struct claim (moves signature) #:transparent)

(define refused (make-weak-hasheq))

;; keyed by the weights rather than kept in the spec, which a layer-copy
;; shares, so a copy's new weights get a lock and a move count of their own
(struct guard (lock moves))

(define guards (make-weak-hasheq))

(define guards-lock (make-lock))

(define (guard-of key)
  (or (hash-ref guards key #f)
      (with-lock guards-lock
        (hash-ref! guards key (lambda () (guard (make-lock) (box 0)))))))

(define (storage-signature weights)
  (for/list ([w (in-list weights)])
    (cons (tensor-data-ptr w) (tensor-storage-ptr w))))

;; Flattening is an optimisation cudnn asks for, never a requirement: a build
;; or a dtype it refuses still runs, on compacted copies, and refusing again
;; on the next call would cost an FFI round trip for the same answer. Three
;; failures are not that and reach the caller: an OOM, which is transient, so
;; the layer stays unflattened and retries once the pressure clears; a
;; contract violation, which is a defect here rather than an answer from
;; cudnn, and would otherwise read as the fallback path; and a native fault.
(define (cudnn-refusal? e)
  (and (exn:fail? e)
       (not (or (exn:fail:rktorch:oom? e)
                (exn:fail:contract? e)
                (native-fault? e)))))

(define (flatten-weights! spec weights)
  (with-handlers ([cudnn-refusal?
                   (lambda (_e) (hash-set! refused (car weights) #t) #f)])
    (with-no-grad
      (cudnn-rnn-flatten-weight weights
                                (if (rnn-bias? spec) 4 2)
                                (rnn-input-size spec)
                                (rnn-cudnn-mode spec)
                                (rnn-hidden-size spec)
                                0
                                (rnn-num-layers spec)
                                (rnn-batch-first? spec)
                                (rnn-bidirectional? spec)))))

(define packing-hook (make-parameter void))

;; a `to` on another thread takes no turn, so record only if no move ran since
;; the signature was read, on any device, since a round trip can hand back the
;; old addresses, and a packing only while the weights still lie in the buffer
(define (packing-record now buffer-start packed moved?)
  (cond
    [moved? #f]
    [(and buffer-start
          (for/and ([entry (in-list packed)]) (= (cdr entry) buffer-start)))
     packed]
    [buffer-start #f]
    [(equal? now packed) now]
    [else #f]))

(define (count-move! weights)
  (void (cas-add! (guard-moves (guard-of (car weights))) 1)))

(define (repack! spec weights)
  ((packing-hook) 'before)
  (hash-remove! refused (car weights))
  (define buffer (flatten-weights! spec weights))
  ((packing-hook) 'after)
  (and buffer (tensor-storage-ptr buffer)))

(define (record-packing! spec weights now moves seen)
  (define key (car weights))
  (define buffer-start
    (and (eq? (device-type (device key)) 'cuda)
         (repack! spec weights)))
  (define record
    (packing-record now
                    buffer-start
                    (storage-signature weights)
                    (not (= seen (unbox moves)))))
  (cond
    [record
     ((packing-hook) 'publish)
     (hash-set! flattened key (claim seen record))]
    [else
     (hash-remove! flattened key)
     (hash-remove! refused key)]))

(define (check-packing! spec weights moves)
  (define seen (unbox moves))
  (define now (storage-signature weights))
  (cond
    [(equal? (claim seen now) (hash-ref flattened (car weights) #f)) (void)]
    [else
     ((packing-hook) 'stale)
     (record-packing! spec weights now moves seen)]))

(define (zero-state spec x)
  (define batch
    (match (tensor-shape x)
      [(list batch-first-dim time-first-dim _)
       (if (rnn-batch-first? spec) batch-first-dim time-first-dim)]))
  (zeros (* (rnn-num-layers spec) (if (rnn-bidirectional? spec) 2 1))
         batch
         (rnn-hidden-size spec)
         #:device (device x)
         #:dtype (dtype x)))

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
  (match-define (guard lock moves) (guard-of (car weights)))
  (with-lock lock (check-packing! spec weights moves))
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
  #:on-move (count-move! (map cdr entries))
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
  #:on-move (count-move! (map cdr entries))
  #:forward (x . state)
  (with-mode (run spec entries gru-input x state mode)))

(module+ private
  (provide count-move! cudnn-refusal? flattened-signature flatten-refused?
           move-count packing-hook packing-lock packing-record
           record-flattened! storage-signature)
  (define (move-count weight) (unbox (guard-moves (guard-of weight))))
  (define (packing-lock weight) (guard-lock (guard-of weight)))
  (define (flattened-signature weight)
    (match (hash-ref flattened weight #f)
      [(claim (== (move-count weight)) signature) signature]
      [_ #f]))
  (define (record-flattened! weight signature)
    (hash-set! flattened weight (claim (move-count weight) signature)))
  (define (flatten-refused? weight) (hash-ref refused weight #f)))

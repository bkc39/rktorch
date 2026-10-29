#lang racket/base

(require (only-in racket/match match match-define)
         (only-in "../../foreign.rkt"
                  * + cat detach device dtype max min reshape tensor->list tensor? to
                  zeros-like))

(provide accumulators
         book-metric
         book-metrics
         epoch-end-reached!
         host-reads
         make-book
         note-epoch!
         note-step!
         read-epoch!
         read-steps!)

(struct logged (logger? both?))

(struct pending logged (value))

(struct entry logged (value weight reduce [cached #:auto]) #:auto-value #f #:mutable)

(struct ledger (steps epoch stepped latest closed))

(struct book (train eval host))

(define (fresh-ledger)
  (ledger (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq)))

(define (make-book) (book (fresh-ledger) (fresh-ledger) (make-hasheq)))

(define reads (box 0))

(define (host-reads) (unbox reads))

(define (loop-ledger b loop)
  (if (eq? loop 'train) (book-train b) (book-eval b)))

(define (scalar v)
  (cond
    [(not (tensor? v)) v]
    [(memq (dtype v) '(float32 float64)) (* (detach v) 1)]
    [else (to (detach v) 'float32)]))

(define (note-step! b loop key v logger? both?)
  (define l (loop-ledger b loop))
  (hash-set! (ledger-steps l) key (pending logger? both? (scalar v)))
  (hash-set! (ledger-stepped l) key #t))

(define (as-tensor v like)
  (if (tensor? v) v (+ (zeros-like like) v)))

(define (larger reduce a b)
  (define pick (if (eq? reduce 'max) max min))
  (if (and (real? a) (real? b))
      (pick a b)
      (pick (cat (list (reshape (as-tensor a b) 1) (reshape (as-tensor b a) 1))))))

(define (combine reduce acc v w)
  (match reduce
    ['mean (+ acc (* v w))]
    ['sum (+ acc v)]
    [_ (larger reduce acc v)]))

(define (start reduce v w)
  (if (eq? reduce 'mean) (* v w) v))

(define (note-epoch! b loop key v w reduce logger? both?)
  (define t (ledger-epoch (loop-ledger b loop)))
  (define value (scalar v))
  (match (hash-ref t key #f)
    [#f (hash-set! t key (entry logger? both? (start reduce value w) w reduce))]
    [e
     (set-entry-value! e (combine reduce (entry-value e) value w))
     (set-entry-weight! e (+ (entry-weight e) w))
     (set-entry-cached! e #f)]))

(define (accumulators b loop)
  (for/hasheq ([(key e) (in-hash (ledger-epoch (loop-ledger b loop)))])
    (values key (entry-value e))))

(define (host-values vs)
  (define ts (filter tensor? vs))
  (define read
    (match ts
      ['() (hasheq)]
      [(cons t _)
       (set-box! reads (add1 (unbox reads)))
       (define dev (device t))
       (define dt (if (memq 'float64 (map dtype ts)) 'float64 'float32))
       (for/hasheq ([t (in-list ts)]
                    [x (in-list (tensor->list
                                 (cat (for/list ([t (in-list ts)])
                                        (reshape (to t dev dt) 1)))))])
         (values t x))]))
  (for/list ([v (in-list vs)])
    (if (tensor? v) (hash-ref read v) v)))

(define (reduced e x)
  (if (eq? (entry-reduce e) 'mean) (exact->inexact (/ x (entry-weight e))) x))

(define (cache! es)
  (define stale (filter (lambda (e) (not (entry-cached e))) es))
  (for ([e (in-list stale)] [x (in-list (host-values (map entry-value stale)))])
    (set-entry-cached! e (reduced e x))))

(define (suffixed key both? suffix)
  (if both? (string->symbol (format "~a-~a" key suffix)) key))

(define (record! b key value both? suffix)
  (define host (book-host b))
  (hash-set! host key value)
  (when both? (hash-set! host (suffixed key both? suffix) value)))

(define (drain! b table host-of suffix [latest #f])
  (define-values (keys records)
    (for/lists (keys records) ([(k r) (in-hash table)]) (values k r)))
  (hash-clear! table)
  (for/fold ([out (hasheq)]) ([k (in-list keys)] [r (in-list records)] [x (in-list (host-of records))])
    (match-define (logged logger? both?) r)
    (record! b k x both? suffix)
    (when latest (hash-set! latest k x))
    (if logger? (hash-set out (suffixed k both? suffix) x) out)))

(define (read-steps! b loop)
  (define l (loop-ledger b loop))
  (drain! b (ledger-steps l)
          (lambda (ps) (host-values (map pending-value ps)))
          "step"
          (ledger-latest l)))

(define (read-epoch! b loop)
  (define l (loop-ledger b loop))
  (drain! b (ledger-epoch l)
          (lambda (es) (cache! es) (map entry-cached es))
          "epoch"
          (ledger-closed l)))

(define (epoch-end-reached! b loop)
  (define l (loop-ledger b loop))
  (for ([key (in-list (hash-keys (ledger-epoch l)))])
    (hash-remove! (ledger-stepped l) key)))

(define (step-metric b l key)
  (match (hash-ref (ledger-steps l) key #f)
    [#f (hash-ref (ledger-latest l) key #f)]
    [(pending logger? both? v)
     (match-define (list x) (host-values (list v)))
     (hash-set! (ledger-steps l) key (pending logger? both? x))
     (record! b key x both? "step")
     x]))

(define (live-metric b l key)
  (define e (hash-ref (ledger-epoch l) key #f))
  (cond
    [(hash-ref (ledger-stepped l) key #f) (step-metric b l key)]
    [e (cache! (list e)) (entry-cached e)]
    [else (hash-ref (ledger-closed l) key #f)]))

(define (book-metric b loop key default)
  (define other (if (eq? loop 'train) 'eval 'train))
  (or (live-metric b (loop-ledger b loop) key)
      (live-metric b (loop-ledger b other) key)
      (hash-ref (book-host b) key default)))

(define (book-metrics b)
  (for/hasheq ([(k v) (in-hash (book-host b))]) (values k v)))

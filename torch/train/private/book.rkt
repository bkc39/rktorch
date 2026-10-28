#lang racket/base

(require (only-in racket/list partition)
         (only-in racket/match match match-define)
         (only-in "../../foreign.rkt"
                  * + cat detach device dtype item max min reshape tensor->list
                  tensor? to))

(provide accumulators
         book-metric
         book-metrics
         host-reads
         make-book
         note-epoch!
         note-step!
         read-epoch!
         read-steps!)

(struct entry (value weight reduce logger? both?) #:mutable)

(struct pending (value logger? both?))

(struct book (steps train eval host [newer-step #:mutable]))

(define (make-book) (book (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq)))

(define reads (box 0))

(define (host-reads) (unbox reads))

(define (scalar v)
  (if (tensor? v) (detach v) v))

(define (note-step! b key v logger? both?)
  (hash-set! (book-steps b) key (pending (scalar v) logger? both?))
  (hash-set! (book-newer-step b) key #t))

(define (larger reduce a b)
  (cond
    [(and (real? a) (real? b)) ((if (eq? reduce 'max) max min) a b)]
    [else
     ((if (eq? reduce 'max) max min)
      (cat (list (reshape (as-tensor a b) 1) (reshape (as-tensor b a) 1))))]))

(define (as-tensor v like)
  (if (tensor? v) v (+ (* (detach like) 0) v)))

(define (combine reduce acc v w)
  (match reduce
    ['mean (+ acc (* v w))]
    ['sum (+ acc v)]
    [_ (larger reduce acc v)]))

(define (start reduce v w)
  (match reduce
    ['mean (* v w)]
    ['sum (* v 1)]
    [_ (* v 1)]))

(define (table b loop)
  (if (eq? loop 'train) (book-train b) (book-eval b)))

(define (note-epoch! b loop key v w reduce logger? both?)
  (define t (table b loop))
  (define value (scalar v))
  (match (hash-ref t key #f)
    [#f (hash-set! t key (entry (start reduce value w) w reduce logger? both?))]
    [e
     (set-entry-value! e (combine reduce (entry-value e) value w))
     (set-entry-weight! e (+ (entry-weight e) w))]))

(define (accumulators b loop)
  (for/hasheq ([(key e) (in-hash (table b loop))])
    (values key (entry-value e))))

(define (host-values vs)
  (define-values (ts rs) (partition tensor? vs))
  (define read
    (cond
      [(null? ts) (hasheq)]
      [else
       (set-box! reads (add1 (unbox reads)))
       (define dev (device (car ts)))
       (define dt (if (memq 'float64 (map dtype ts)) 'float64 'float32))
       (for/hasheq ([t (in-list ts)]
                    [x (in-list (tensor->list
                                 (cat (for/list ([t (in-list ts)])
                                        (reshape (to t dev dt) 1)))))])
         (values t x))]))
  (for/list ([v (in-list vs)])
    (if (tensor? v) (hash-ref read v) v)))

(define (suffixed key both? suffix)
  (if both? (string->symbol (format "~a-~a" key suffix)) key))

(define (record! b key value both? suffix)
  (define host (book-host b))
  (hash-set! host key value)
  (when both? (hash-set! host (suffixed key both? suffix) value)))

(define (read-steps! b)
  (define steps (book-steps b))
  (define keys (hash-keys steps))
  (define values-read (host-values (for/list ([k (in-list keys)]) (pending-value (hash-ref steps k)))))
  (define logged
    (for/fold ([logged (hasheq)]) ([k (in-list keys)] [x (in-list values-read)])
      (match-define (pending _ logger? both?) (hash-ref steps k))
      (record! b k x both? "step")
      (hash-remove! (book-newer-step b) k)
      (if logger? (hash-set logged (suffixed k both? "step") x) logged)))
  (hash-clear! steps)
  logged)

(define (reduced e x)
  (if (eq? (entry-reduce e) 'mean) (exact->inexact (/ x (entry-weight e))) x))

(define (read-epoch! b loop)
  (define t (table b loop))
  (define keys (hash-keys t))
  (define values-read (host-values (for/list ([k (in-list keys)]) (entry-value (hash-ref t k)))))
  (define logged
    (for/fold ([logged (hasheq)]) ([k (in-list keys)] [x (in-list values-read)])
      (define e (hash-ref t k))
      (define value (reduced e x))
      (record! b k value (entry-both? e) "epoch")
      (hash-remove! (book-newer-step b) k)
      (if (entry-logger? e) (hash-set logged (suffixed k (entry-both? e) "epoch") value) logged)))
  (hash-clear! t)
  logged)

(define (book-metric b key default)
  (match (and (hash-ref (book-newer-step b) key #f) (hash-ref (book-steps b) key #f))
    [(pending v _ both?)
     (define x (if (tensor? v) (item v) v))
     (record! b key x both? "step")
     (hash-remove! (book-newer-step b) key)
     x]
    [#f (hash-ref (book-host b) key default)]))

(define (book-metrics b)
  (for/hasheq ([(k v) (in-hash (book-host b))]) (values k v)))

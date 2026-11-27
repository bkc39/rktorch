#lang racket/base

(require (only-in racket/dict in-dict in-dict-keys)
         (only-in racket/function curry)
         (only-in racket/list check-duplicates remove-duplicates)
         (only-in racket/match match match-define)
         (only-in threading ~>)
         (only-in "../../foreign.rkt"
                  * + cat detach device dtype max min numel reshape tensor->list tensor? to
                  zeros-like))

(provide accumulators
         book-metric
         book-metrics
         book-state
         book-writes
         epoch-end-reached!
         host-reads
         make-book
         note-epoch!
         note-step!
         read-epoch!
         read-groups
         read-steps!
         restore-book-state!
         saved-book-checks)

(struct logged (logger? both?))

(struct pending logged (value))

(struct entry logged (value weight reduce [cached #:auto]) #:auto-value #f #:mutable)

(struct ledger (steps epoch stepped latest closed))

(struct book (train eval host [writes #:mutable]))

(define (fresh-ledger)
  (ledger (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq)))

(define (make-book) (book (fresh-ledger) (fresh-ledger) (make-hasheq) 0))

(define (count-write! b)
  (set-book-writes! b (add1 (book-writes b))))

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
  (count-write! b)
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
  (count-write! b)
  (match (hash-ref t key #f)
    [#f (hash-set! t key (entry logger? both? (start reduce value w) w reduce))]
    [e
     (set-entry-value! e (combine reduce (entry-value e) value w))
     (set-entry-weight! e (+ (entry-weight e) w))
     (set-entry-cached! e #f)]))

(define (accumulators b loop)
  (for/hasheq ([(key e) (in-hash (ledger-epoch (loop-ledger b loop)))])
    (values key (entry-value e))))

(define (read-groups ts #:device [device-of device] #:dtype [dtype-of dtype])
  (for/list ([dev (in-list (remove-duplicates (map device-of ts)))])
    (define members (filter (lambda (t) (equal? (device-of t) dev)) ts))
    (list dev (if (memq 'float64 (map dtype-of members)) 'float64 'float32) members)))

(define (read-group dev dt members)
  (set-box! reads (add1 (unbox reads)))
  (~> (for/list ([t (in-list members)]) (reshape (to t dev dt) 1)) cat tensor->list))

(define (host-values vs)
  (define read
    (for/fold ([read (hasheq)]) ([g (in-list (read-groups (filter tensor? vs)))])
      (match-define (list dev dt members) g)
      (for/fold ([read read]) ([t (in-list members)] [x (in-list (read-group dev dt members))])
        (hash-set read t x))))
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

(struct record-shape (slots make))

(struct slot (get ok?))

(define reducers '(mean sum max min))

(define (metric-value? v) (or (real? v) (and (tensor? v) (= (numel v) 1))))

(define (weight? w) (and (real? w) (not (negative? w))))

(define (reducer? r) (and (memq r reducers) #t))

(define logged-slots
  (list (cons "logger" (slot logged-logger? boolean?)) (cons "both" (slot logged-both? boolean?))))

(define pending-shape
  (record-shape (append logged-slots (list (cons "value" (slot pending-value metric-value?))))
                pending))

(define entry-shape
  (record-shape (append logged-slots
                        (list (cons "value" (slot entry-value metric-value?))
                              (cons "weight" (slot entry-weight weight?))
                              (cons "reduce" (slot entry-reduce reducer?))))
                entry))

(define ledger-fields
  (list (list "steps" ledger-steps pending-shape)
        (list "epoch" ledger-epoch entry-shape)
        (list "stepped" ledger-stepped boolean?)
        (list "latest" ledger-latest real?)
        (list "closed" ledger-closed real?)))

(define (shape-state shape v)
  (match shape
    [(record-shape slots _)
     (for/hash ([(name s) (in-dict slots)]) (values name ((slot-get s) v)))]
    [_ v]))

(define (state-shape shape saved)
  (match shape
    [(record-shape slots make)
     (apply make (for/list ([name (in-dict-keys slots)]) (hash-ref saved name)))]
    [_ saved]))

(define (shape-state? shape saved)
  (match shape
    [(record-shape slots _)
     (and (hash? saved)
          (for/and ([(name s) (in-dict slots)])
            (and (hash-has-key? saved name) ((slot-ok? s) (hash-ref saved name)))))]
    [ok? (ok? saved)]))

(define (string-keyed-state table key-path state-of)
  (define names (hash-keys table))
  (match (check-duplicates names #:key symbol->string)
    [#f (for/hash ([(k v) (in-hash table)]) (values (symbol->string k) (state-of v)))]
    [twin
     (define twins (filter (lambda (k) (equal? (symbol->string k) (symbol->string twin))) names))
     (raise-arguments-error
      'save-checkpoint!
      "two logged metric names are spelled the same, so the checkpoint cannot hold both"
      "key" key-path
      "names" twins
      "interned" (map symbol-interned? twins))]))

(define (ledger-state l loop)
  (for/hash ([field (in-list ledger-fields)])
    (match-define (list name table-of shape) field)
    (values name (string-keyed-state (table-of l) (format "metric_book.~a.~a" loop name)
                                     (curry shape-state shape)))))

(define (book-state b)
  (hash "train" (ledger-state (book-train b) "train") "eval" (ledger-state (book-eval b) "eval")))

(define ((saved-table? shape) table)
  (and (hash? table)
       (for/and ([(k v) (in-hash table)]) (and (string? k) (shape-state? shape v)))))

(define saved-book-checks
  (for*/list ([loop (in-list '("train" "eval"))] [field (in-list ledger-fields)])
    (match-define (list name _ shape) field)
    (cons (list loop name) (saved-table? shape))))

(define (refill! table saved [load values])
  (hash-clear! table)
  (for ([(k v) (in-hash saved)])
    (hash-set! table (string->symbol k) (load v))))

(define (restore-ledger! l saved)
  (for ([field (in-list ledger-fields)])
    (match-define (list name table-of shape) field)
    (refill! (table-of l) (hash-ref saved name) (curry state-shape shape))))

(define (restore-book-state! b metrics saved #:close-train-epoch? close?)
  (refill! (book-host b) metrics)
  (when saved
    (restore-ledger! (book-train b) (hash-ref saved "train"))
    (restore-ledger! (book-eval b) (hash-ref saved "eval")))
  (when close?
    (void (read-epoch! b 'train))))

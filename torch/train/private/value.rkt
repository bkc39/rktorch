#lang racket/base

(require (only-in racket/match match)
         (only-in "../../foreign.rkt" dtype tensor?))

(provide checkpoint-value?
         decode-value
         encode-value
         value-problem)

(define tagged-dtypes '(float32 float64 float16 bfloat16 int64 int32 int16 int8 uint8 bool))

(define (reserved-key? k)
  (and (string? k) (positive? (string-length k)) (char=? (string-ref k 0) #\$)))

(define (exact-rational? x)
  (and (rational? x) (exact? x)))

(define (taggable-hash? h)
  (and (or (hash-equal? h) (hash-eqv? h) (hash-eq? h))
       (not (hash-weak? h))
       (not (hash-ephemeron? h))))

(define (value-problem v)
  (define on-path (make-hasheq))
  (define accepted (make-hasheq))
  (define (node v path walk)
    (cond
      [(hash-ref accepted v #f) #f]
      [(hash-ref on-path v #f) (cons path v)]
      [else
       (hash-set! on-path v #t)
       (define problem (walk))
       (hash-remove! on-path v)
       (unless problem (hash-set! accepted v #t))
       problem]))
  (define (bad path v) (cons path v))
  (define (spine p i path)
    (node p path
          (lambda ()
            (or (check (car p) (cons i path))
                (match (cdr p)
                  [(? pair? d) (spine d (add1 i) path)]
                  [d (check d (cons 'tail path))])))))
  (define (check v path)
    (match v
      [(or #t #f '()) #f]
      [(or (? flonum?) (? string?) (? exact-rational?)) #f]
      [(? symbol?) (and (not (symbol-interned? v)) (bad path v))]
      [(? tensor?) (and (not (memq (dtype v) tagged-dtypes)) (bad path v))]
      [(? pair?) (spine v 0 path)]
      [(? hash?)
       #:when (taggable-hash? v)
       (node v path
             (lambda ()
               (for/or ([(k x) (in-hash v)])
                 (cond
                   [(reserved-key? k) (bad (cons k path) k)]
                   [else (or (check k (cons k path)) (check x (cons k path)))]))))]
      [_ (bad path v)]))
  (match (check v '())
    [#f #f]
    [(cons path part) (cons (reverse path) part)]))

(define (checkpoint-value? v)
  (not (value-problem v)))

(define (hash-kind h)
  (cond
    [(hash-eq? h) "eq"]
    [(hash-eqv? h) "eqv"]
    [else "equal"]))

(define (object-hash? h)
  (and (hash-equal? h) (for/and ([k (in-hash-keys h)]) (string? k))))

;; entries are visited in a fixed order so tensors are numbered the same way
;; on every save of an equal tree
(define (order-key k)
  (format "~s" (let strip ([k k])
                 (match k
                   [(? tensor?) '|#<tensor>|]
                   [(cons a d) (cons (strip a) (strip d))]
                   [_ k]))))

(define (special-float x)
  (cond
    [(eqv? x +inf.0) "+inf.0"]
    [(eqv? x -inf.0) "-inf.0"]
    [(eqv? x +nan.0) "+nan.0"]
    [else #f]))

(define (encode-value v name-tensor #:path [path '()])
  (let encode ([v v] [path path])
    (match v
      [(or #t #f (? string?) (? exact-integer?) '()) v]
      [(? exact-rational?) (hasheq '$rational (number->string v))]
      [(? flonum?)
       (match (special-float v)
         [#f v]
         [s (hasheq '$float s)])]
      [(? symbol?) (hasheq '$symbol (symbol->string v))]
      [(? tensor?) (hasheq '$tensor (name-tensor v (reverse path)))]
      [(? list?) (for/list ([x (in-list v)] [i (in-naturals)]) (encode x (cons i path)))]
      [(cons a d) (hasheq '$pair (list (encode a (cons 0 path)) (encode d (cons 1 path))))]
      [(? object-hash?)
       (for/hasheq ([k (in-list (sort (hash-keys v) string<?))])
         (values (string->symbol k) (encode (hash-ref v k) (cons k path))))]
      [(? hash?)
       (hasheq '$hash (hash-kind v)
               'entries
               (for/list ([k (in-list (sort (hash-keys v) string<? #:key order-key
                                            #:cache-keys? #t))])
                 (list (encode k (cons k path)) (encode (hash-ref v k) (cons k path)))))])))

(define (malformed j)
  (raise-arguments-error 'read-checkpoint "malformed checkpoint value" "value" j))

(define (decode-value j tensor-named)
  (let decode ([j j])
    (match j
      [(? list?) (map decode j)]
      [(? hash?)
       (match (hash-keys j)
         [(list '$tensor) (tensor-named (hash-ref j '$tensor))]
         [(list '$symbol) (string->symbol (hash-ref j '$symbol))]
         [(list '$float) (match (hash-ref j '$float)
                           ["+inf.0" +inf.0]
                           ["-inf.0" -inf.0]
                           ["+nan.0" +nan.0]
                           [_ (malformed j)])]
         [(list '$rational)
          (match (string->number (hash-ref j '$rational) 10)
            [(? exact-rational? q) q]
            [_ (malformed j)])]
         [(list '$pair)
          (match (hash-ref j '$pair)
            [(list a d) (cons (decode a) (decode d))]
            [_ (malformed j)])]
         [(or (list '$hash 'entries) (list 'entries '$hash))
          (define pairs
            (for/list ([e (in-list (hash-ref j 'entries))])
              (match e
                [(list k v) (cons (decode k) (decode v))]
                [_ (malformed j)])))
          (match (hash-ref j '$hash)
            ["eq" (make-immutable-hasheq pairs)]
            ["eqv" (make-immutable-hasheqv pairs)]
            ["equal" (make-immutable-hash pairs)]
            [_ (malformed j)])]
         [keys
          (for/hash ([k (in-list keys)])
            (define s (symbol->string k))
            (when (reserved-key? s) (malformed j))
            (values s (decode (hash-ref j k))))])]
      [_ j])))

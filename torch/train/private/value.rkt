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

(define (identity-survives-read? k)
  (or (boolean? k) (null? k) (fixnum? k) (and (symbol? k) (symbol-interned? k))))

(define (key-survives-read? h k)
  (cond
    [(hash-eq? h) (identity-survives-read? k)]
    [(hash-eqv? h) (or (identity-survives-read? k) (flonum? k) (exact-rational? k))]
    [else #t]))

(define (value-problem v)
  (define on-path (make-hasheq))
  (define accepted (make-hasheq))
  (define identity-bound (make-hasheq))
  (define (bound-to-identity? k)
    (match k
      [(? tensor?) #t]
      [(or (? pair?) (? hash?))
       (hash-ref! identity-bound k
                  (lambda ()
                    (match k
                      [(cons a d) (or (bound-to-identity? a) (bound-to-identity? d))]
                      [_ (or (not (immutable? k))
                             (for/or ([(a x) (in-hash k)])
                               (or (bound-to-identity? a) (bound-to-identity? x))))])))]
      [_ #f]))
  (define (bad path v) (cons path v))
  (define (node v path walk)
    (cond
      [(hash-ref accepted v #f) #f]
      [(hash-ref on-path v #f) (bad path v)]
      [else
       (hash-set! on-path v #t)
       (define problem (walk))
       (hash-remove! on-path v)
       (unless problem (hash-set! accepted v #t))
       problem]))
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
                 (define at (cons k path))
                 (if (or (reserved-key? k) (not (key-survives-read? v k)))
                     (bad at k)
                     (or (check k at)
                         (and (bound-to-identity? k) (bad at k))
                         (check x at))))))]
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

(define (keys-in-a-fixed-order h)
  (define (printed k)
    (format "~s" (let strip ([k k])
                   (match k
                     [(? tensor?) '|#<tensor>|]
                     [(cons a d) (cons (strip a) (strip d))]
                     [_ k]))))
  (sort (hash-keys h) string<? #:key printed #:cache-keys? #t))

(define (special-float? x)
  (or (eqv? x -0.0) (not (< -inf.0 x +inf.0))))

(define (encode-value v name-tensor)
  (define encoded (make-hasheq))
  (let encode ([v v])
    (match v
      [(or #t #f (? string?) (? exact-integer?) '()) v]
      [(? exact-rational?) (hasheq '$rational (number->string v))]
      [(? flonum?) (if (special-float? v) (hasheq '$float (number->string v)) v)]
      [(? symbol?) (hasheq '$symbol (symbol->string v))]
      [(? tensor?) (hasheq '$tensor (name-tensor v))]
      [_
       (hash-ref! encoded v
                  (lambda ()
                    (match v
                      [(? list?) (map encode v)]
                      [(cons a d) (hasheq '$pair (list (encode a) (encode d)))]
                      [(? object-hash?)
                       (for/hasheq ([k (in-list (sort (hash-keys v) string<?))])
                         (values (string->symbol k) (encode (hash-ref v k))))]
                      [_
                       (hasheq '$hash (hash-kind v)
                               'entries (for/list ([k (in-list (keys-in-a-fixed-order v))])
                                          (list (encode k) (encode (hash-ref v k)))))])))])))

(define (malformed j)
  (raise-arguments-error 'read-checkpoint "malformed checkpoint value" "value" j))

(define (tagged-number j tag ok?)
  (match (hash-ref j tag)
    [(? string? s)
     (match (string->number s 10)
       [(? ok? x) x]
       [_ (malformed j)])]
    [_ (malformed j)]))

(define (decode-value j tensor-named)
  (let decode ([j j])
    (match j
      [(or #t #f (? string?) (? exact-integer?) (? flonum?)) j]
      [(? list?) (map decode j)]
      [(? hash?)
       (match (hash-keys j)
         [(list '$tensor) (tensor-named (hash-ref j '$tensor))]
         [(list '$symbol) (string->symbol (hash-ref j '$symbol))]
         [(list '$float)
          (tagged-number j '$float (lambda (x) (and (flonum? x) (special-float? x))))]
         [(list '$rational) (tagged-number j '$rational exact-rational?)]
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
      [_ (malformed j)])))

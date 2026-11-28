#lang racket/base

(require (only-in racket/list check-duplicates)
         (only-in racket/match match)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "../../foreign/raw/tensor.rkt" dtype-code->symbol tr-tensor-dtype/raw))

(provide checkpoint-value?
         decode-value
         encode-value
         string-keyed-state
         string-keyed?
         value-problem)

(define tagged-dtypes '(float32 float64 float16 bfloat16 int64 uint8 bool))

(define (tagged-dtype? t)
  (define-values (rc code) (tr-tensor-dtype/raw t))
  (and (zero? rc) (memq (dtype-code->symbol code) tagged-dtypes) #t))

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
      [(cons a d)
       (hash-ref! identity-bound k (lambda () (or (bound-to-identity? a) (bound-to-identity? d))))]
      [(? hash?) (hash-ref! identity-bound k (lambda () (hash-bound-to-identity? k)))]
      [_ #f]))
  (define (hash-bound-to-identity? h)
    (or (not (immutable? h))
        (for/or ([(a x) (in-hash h)])
          (or (bound-to-identity? a) (bound-to-identity? x)))))
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
      [(? tensor?) (and (not (tagged-dtype? v)) (bad path v))]
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

(define (string-keyed-state table key-path [state-of values])
  (define names (hash-keys table))
  (match (check-duplicates names #:key symbol->string)
    [#f (for/hash ([(k v) (in-hash table)]) (values (symbol->string k) (state-of v)))]
    [twin
     (define twins (filter (lambda (k) (equal? (symbol->string k) (symbol->string twin))) names))
     (raise-arguments-error
      'save-checkpoint!
      "two names are spelled the same, so the checkpoint cannot hold both"
      "key" key-path
      "names" twins
      "interned" (map symbol-interned? twins))]))

(define (string-keyed? v)
  (and (hash? v) (for/and ([k (in-hash-keys v)]) (string? k))))

(define (object-hash? h)
  (and (hash-equal? h) (string-keyed? h)))

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

(define (decode-value j tensor-named malformed)
  (define (payload tagged tag ok?)
    (match (hash-ref tagged tag)
      [(? ok? x) x]
      [_ (malformed)]))
  (define (tagged-number tagged tag ok?)
    (match (string->number (payload tagged tag string?) 10)
      [(? ok? x) x]
      [_ (malformed)]))
  (let decode ([j j])
    (match j
      [(or #t #f (? string?) (? exact-integer?) (? flonum?)) j]
      [(? list?) (map decode j)]
      [(? hash?)
       (match (hash-keys j)
         [(list '$tensor) (tensor-named (payload j '$tensor string?))]
         [(list '$symbol) (string->symbol (payload j '$symbol string?))]
         [(list '$float)
          (tagged-number j '$float (lambda (x) (and (flonum? x) (special-float? x))))]
         [(list '$rational) (tagged-number j '$rational exact-rational?)]
         [(list '$pair)
          (match (hash-ref j '$pair)
            [(list a d) (cons (decode a) (decode d))]
            [_ (malformed)])]
         [(or (list '$hash 'entries) (list 'entries '$hash))
          (define pairs
            (for/list ([e (in-list (payload j 'entries list?))])
              (match e
                [(list k v) (cons (decode k) (decode v))]
                [_ (malformed)])))
          (match (hash-ref j '$hash)
            ["eq" (make-immutable-hasheq pairs)]
            ["eqv" (make-immutable-hasheqv pairs)]
            ["equal" (make-immutable-hash pairs)]
            [_ (malformed)])]
         [keys
          (for/hash ([k (in-list keys)])
            (define s (symbol->string k))
            (when (reserved-key? s) (malformed))
            (values s (decode (hash-ref j k))))])]
      [_ (malformed)])))

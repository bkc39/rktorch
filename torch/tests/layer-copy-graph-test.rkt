#lang racket/base

(module+ test
  (require (only-in racket/dict in-dict in-dict-keys)
           (only-in racket/list shuffle)
           (only-in racket/match match match* match-define match-lambda)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/copy-layers.rkt" twin twin?)
           (only-in "private/within-memory.rkt" within-memory))

  (struct graph (root nodes))

  (define atoms '(0 1 a b "s"))
  (define atom-keys '(k1 k2 k3 4 5))

  (define (pick xs)
    (list-ref xs (random (length xs))))

  (define (key-id k)
    (match k
      [(or (cons (? exact-integer? id) _) (vector (? exact-integer? id) _))
       #:when (>= id 1000)
       id]
      [_ #f]))

  (define (mutable-key id)
    (vector id #f))

  (define (immutable-key id payload)
    (if (zero? (random 2)) (cons id payload) (vector-immutable id payload)))

  (define fresh-mutables
    (list (list 'box (lambda () (box #f)))
          (list 'vector (lambda () (make-vector (random 1 4) #f)))
          (list 'mpair (lambda () (mcons #f #f)))
          (list 'hasheq make-hasheq)
          (list 'hash make-hash)
          (list 'hashalw make-hashalw)))

  (define (table-keys kind keys)
    (match* (kind keys)
      [('ihash (cons _ immutable-keys)) immutable-keys]
      [(_ (cons mutable-keys immutable-keys)) (append mutable-keys immutable-keys)]))

  (define (hash-key keys)
    (if (and (pair? keys) (zero? (random 2))) (pick keys) (pick atom-keys)))

  (define (entries kind keys refs)
    (for/list ([_ (in-range (random 0 4))])
      (cons (hash-key (table-keys kind keys)) (pick refs))))

  (define (immutable-of kind refs keys)
    (define (ref) (pick refs))
    (define (table make) (make (entries kind keys refs)))
    (match kind
      ['pair (cons (ref) (ref))]
      ['ivector (apply vector-immutable (for/list ([_ (in-range (random 1 4))]) (ref)))]
      ['ibox (box-immutable (ref))]
      ['ihasheq (table make-immutable-hasheq)]
      ['ihash (table make-immutable-hash)]
      ['ihashalw (table make-immutable-hashalw)]))

  (define (fill! m kind refs keys)
    (define (ref) (pick refs))
    (match m
      [(? mpair?) (set-mcar! m (ref)) (set-mcdr! m (ref))]
      [(? box?) (set-box! m (ref))]
      [(? vector?) (for ([i (in-range (vector-length m))]) (vector-set! m i (ref)))]
      [(? hash?)
       (for ([(k x) (in-dict (entries kind keys refs))])
         (hash-set! m k x))]))

  (define (random-graph)
    (define w (Parameter (ones 1)))
    (define kinds (for/list ([_ (in-range (random 1 6))]) (pick fresh-mutables)))
    (define mutables (map (match-lambda [(list _ make) (make)]) kinds))
    (define mutable-keys (for/list ([i (in-range (random 0 3))]) (mutable-key (+ 1000 i))))
    (define (layers-over refs n)
      (for/fold ([made '()]) ([_ (in-range n)])
        (define pool (append refs made))
        (cons (twin (pick pool) (pick pool)) made)))
    (define early (layers-over (append (list w 0) mutables mutable-keys) (random 1 3)))
    (define immutable-keys
      (for/list ([i (in-range (random 0 3))])
        (immutable-key (+ 2000 i) (pick (append atoms early)))))
    (define keys (cons mutable-keys immutable-keys))
    (define base (append (list w 0 'a) mutables mutable-keys immutable-keys early))
    (define immutables
      (for/fold ([made '()]) ([_ (in-range (random 1 6))])
        (define kind (pick '(pair ivector ibox ihasheq ihash ihashalw)))
        (cons (immutable-of kind (append atoms base made) keys) made)))
    (define late (layers-over (append base immutables) (random 0 3)))
    (define everything (append base immutables late))
    (for ([k (in-list mutable-keys)])
      (vector-set! k 1 (pick (append (list 0) mutable-keys early late))))
    (for ([m (in-list mutables)] [k (in-list kinds)])
      (match-define (list kind _) k)
      (fill! m kind (append (list 1 'b) everything) keys))
    (define nodes (append mutables mutable-keys immutable-keys early immutables late))
    (~> (shuffle nodes) (twin 0) (graph nodes)))

  (define (atomic? v)
    (or (number? v) (symbol? v) (string? v) (null? v) (boolean? v)))

  (define (fresh? v)
    (or (twin? v) (tensor? v) (mpair? v)
        (and (or (box? v) (vector? v) (hash? v)) (not (immutable? v)))))

  (define (same-kind? o c)
    (cond
      [(pair? o) (pair? c)]
      [(mpair? o) (mpair? c)]
      [(box? o) (and (box? c) (eq? (immutable? o) (immutable? c)))]
      [(vector? o) (and (vector? c) (eq? (immutable? o) (immutable? c))
                        (= (vector-length o) (vector-length c)))]
      [(hash? o) (and (hash? c) (eq? (immutable? o) (immutable? c))
                      (eq? (hash-equal? o) (hash-equal? c)) (eq? (hash-eq? o) (hash-eq? c))
                      (eq? (hash-equal-always? o) (hash-equal-always? c))
                      (= (hash-count o) (hash-count c)))]
      [(twin? o) (twin? c)]
      [(tensor? o) (tensor? c)]
      [else #f]))

  (define (copy-key c k)
    (define id (key-id k))
    (if id
        (for/first ([ck (in-hash-keys c)] #:when (eqv? (key-id ck) id))
          ck)
        k))

  (define (paired o c)
    (match* (o c)
      [((cons a d) (cons a2 d2)) (list (cons a a2) (cons d d2))]
      [((mcons a d) (mcons a2 d2)) (list (cons a a2) (cons d d2))]
      [((box x) (box y)) (list (cons x y))]
      [((? vector?) _) (for/list ([x (in-vector o)] [y (in-vector c)]) (cons x y))]
      [((? hash?) _)
       (for*/list ([(k x) (in-hash o)]
                   [k2 (in-value (copy-key c k))]
                   [part (in-list (list (cons k k2) (cons x (hash-ref c k2 'missing))))])
         part)]
      [((? twin?) _) (map cons (o) (c))]
      [(_ _) '()]))

  (define (reaches-fresh? v)
    (let visit ([v v] [seen (hasheq)])
      (and (not (hash-ref seen v #f))
           (or (fresh? v)
               (for/or ([part (in-dict-keys (paired v v))])
                 (visit part (hash-set seen v #t)))))))

  (define (image-map orig copy)
    (define φ (make-hasheq))
    (define ψ (make-hasheq))
    (let walk ([o orig] [c copy])
      (cond
        [(atomic? o) (check-equal? c o)]
        [(hash-ref φ o #f) => (lambda (c0) (check-eq? c c0 "φ is a function"))]
        [else
         (check-false (hash-ref ψ c #f) "φ is injective")
         (check-true (same-kind? o c) (format "kind of ~a" o))
         (hash-set! φ o c)
         (hash-set! ψ c o)
         (for ([(o2 c2) (in-dict (paired o c))])
           (walk o2 c2))]))
    φ)

  (define (check-registry φ)
    (for ([(o c) (in-hash φ)] #:when (twin? o))
      (check-equal? (map car (layer-named-children c)) (map car (layer-named-children o)))
      (check-equal? (map cdr (layer-named-children c))
                    (map (match-lambda [(cons _ child) (hash-ref φ child)])
                         (layer-named-children o)))
      (check-equal? (parameters c) (map (lambda (t) (hash-ref φ t)) (parameters o)))))

  (define (check-isomorphic g)
    (define root (graph-root g))
    (define copy (layer-copy root))
    (define φ (image-map root copy))
    (check-registry φ)
    (for ([(o c) (in-hash φ)])
      (cond
        [(fresh? o) (check-false (hash-has-key? φ c) "no fresh copy is a node of the original")]
        [(reaches-fresh? o) (check-false (eq? c o) "an immutable over a fresh value is rebuilt")]
        [else (check-eq? c o "an immutable over no fresh value is shared")]))
    (for ([n (in-list (graph-nodes g))])
      (check-true (hash-has-key? φ n) "every node is copied")))

  (test-case "layer-copy is an isomorphism of random cyclic graphs (seeded, 300 graphs)"
    (within-memory
     512
     (lambda ()
       (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
         (random-seed 448)
         (for ([_ (in-range 300)])
           (check-isomorphic (random-graph)))))
     #:seconds 120)))

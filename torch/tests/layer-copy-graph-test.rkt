#lang racket/base

(module+ test
  (require (only-in racket/list shuffle)
           (only-in racket/match match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/copy-layers.rkt" twin twin?)
           (only-in "private/within-memory.rkt" within-memory))

  (struct graph (root nodes))

  (define (pick xs)
    (list-ref xs (random (length xs))))

  (define (key-node id)
    (vector id #f))

  (define (key-node? v)
    (and (vector? v) (not (immutable? v)) (= (vector-length v) 2) (exact-integer? (vector-ref v 0))
         (>= (vector-ref v 0) 1000)))

  (define (fresh-mutable)
    (match (pick '(box vector mpair hasheq hash))
      ['box (box #f)]
      ['vector (make-vector (random 1 4) #f)]
      ['mpair (mcons #f #f)]
      ['hasheq (make-hasheq)]
      ['hash (make-hash)]))

  (define (atom)
    (pick '(0 1 a b "s")))

  (define (hash-key keys)
    (if (and (pair? keys) (zero? (random 2))) (pick keys) (pick '(k1 k2 k3 4 5))))

  (define (immutable-content kind refs ph pair-ph keys)
    (define back (if (eq? kind 'pair) ph pair-ph))
    (define (ref) (if (and (pair? back) (zero? (random 3))) (pick back) (pick refs)))
    (match kind
      ['pair (cons (ref) (ref))]
      ['ivector (apply vector-immutable (for/list ([_ (in-range (random 1 4))]) (ref)))]
      ['ibox (box-immutable (ref))]
      ['ihasheq (for/fold ([h (hasheq)]) ([_ (in-range (random 1 3))])
                  (hash-set h (hash-key keys) (ref)))]
      ['ihash (for/fold ([h (hash)]) ([_ (in-range (random 1 3))])
                (hash-set h (pick '(k1 k2 k3 4 5)) (ref)))]))

  (define (fill! m refs keys)
    (define (ref) (pick refs))
    (match m
      [(? mpair?) (set-mcar! m (ref)) (set-mcdr! m (ref))]
      [(? box?) (set-box! m (ref))]
      [(? vector?) (for ([i (in-range (vector-length m))]) (vector-set! m i (ref)))]
      [(? hash?)
       (for ([_ (in-range (random 0 4))])
         (hash-set! m (hash-key keys) (ref)))]))

  (define (random-graph)
    (define w (Parameter (ones 1)))
    (define mutables (for/list ([_ (in-range (random 1 6))]) (fresh-mutable)))
    (define keys (for/list ([i (in-range (random 0 3))]) (key-node (+ 1000 i))))
    (define (layers-over refs n)
      (for/fold ([made '()]) ([_ (in-range n)])
        (define pool (append refs made))
        (cons (twin (pick pool) (pick pool)) made)))
    (define early (layers-over (append (list w 0) mutables keys) (random 1 3)))
    (define kinds (for/list ([_ (in-range (random 1 5))]) (pick '(pair ivector ibox ihasheq ihash))))
    (define ph (for/list ([_ (in-list kinds)]) (make-placeholder #f)))
    (define pair-ph (for/list ([p (in-list ph)] [k (in-list kinds)] #:when (eq? k 'pair)) p))
    (define base (append (list w 0 'a) mutables keys early))
    (for ([p (in-list ph)] [k (in-list kinds)])
      (placeholder-set! p (immutable-content k base ph pair-ph keys)))
    (define immutables (make-reader-graph ph))
    (define late (layers-over (append base immutables) (random 0 3)))
    (define everything (append base immutables late))
    (for ([k (in-list keys)])
      (vector-set! k 1 (pick (append (list 0) keys early late))))
    (for ([m (in-list mutables)])
      (fill! m (append (list 1 'b) everything) keys))
    (define nodes (append mutables keys early immutables late))
    (graph (twin (shuffle nodes) 0) nodes))

  (define (atomic? v)
    (or (number? v) (symbol? v) (string? v) (null? v) (boolean? v)))

  (define (same-kind? o c)
    (cond
      [(pair? o) (pair? c)]
      [(mpair? o) (mpair? c)]
      [(box? o) (and (box? c) (eq? (immutable? o) (immutable? c)))]
      [(vector? o) (and (vector? c) (eq? (immutable? o) (immutable? c))
                        (= (vector-length o) (vector-length c)))]
      [(hash? o) (and (hash? c) (eq? (immutable? o) (immutable? c))
                      (eq? (hash-equal? o) (hash-equal? c)) (eq? (hash-eq? o) (hash-eq? c))
                      (= (hash-count o) (hash-count c)))]
      [(twin? o) (twin? c)]
      [(tensor? o) (tensor? c)]
      [else #f]))

  (define (copy-key c k)
    (if (key-node? k)
        (for/first ([ck (in-hash-keys c)]
                    #:when (and (key-node? ck) (= (vector-ref ck 0) (vector-ref k 0))))
          ck)
        k))

  (define (paired o c)
    (match o
      [(cons a d) (list (cons a (car c)) (cons d (cdr c)))]
      [(mcons a d) (list (cons a (mcar c)) (cons d (mcdr c)))]
      [(box x) (list (cons x (unbox c)))]
      [(? vector?) (for/list ([x (in-vector o)] [y (in-vector c)]) (cons x y))]
      [(? hash?)
       (for*/list ([(k x) (in-hash o)]
                   [k2 (in-value (copy-key c k))]
                   [part (in-list (list (cons k k2) (cons x (hash-ref c k2 'missing))))])
         part)]
      [(? twin?) (map cons (o) (c))]
      [_ '()]))

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
         (for ([oc (in-list (paired o c))])
           (walk (car oc) (cdr oc)))]))
    φ)

  (define (check-registry φ)
    (for ([(o c) (in-hash φ)] #:when (twin? o))
      (check-equal? (map car (layer-named-children c)) (map car (layer-named-children o)))
      (check-equal? (map cdr (layer-named-children c))
                    (map (lambda (e) (hash-ref φ (cdr e))) (layer-named-children o)))
      (check-equal? (parameters c) (map (lambda (t) (hash-ref φ t)) (parameters o)))))

  (define (check-isomorphic g)
    (define root (graph-root g))
    (define copy (layer-copy root))
    (define φ (image-map root copy))
    (check-registry φ)
    (for ([(o c) (in-hash φ)])
      (check-false (hash-has-key? φ c) "no copy is a node of the original"))
    (for ([n (in-list (graph-nodes g))])
      (check-true (hash-has-key? φ n) "every node is copied")))

  (test-case "layer-copy is an isomorphism of random cyclic graphs (seeded, 200 graphs)"
    (within-memory
     512
     (lambda ()
       (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
         (random-seed 448)
         (for ([_ (in-range 200)])
           (check-isomorphic (random-graph)))))
     #:seconds 120)))
